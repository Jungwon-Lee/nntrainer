# KV Cache Optimizer 구조 변경 계획

## 배경

현재 CausalLM의 KV cache는 `KVCacheManager`가 메모리를 소유하고,
`mha_core`가 그 메모리를 external input tensor로 받아 직접 읽고 쓰는 구조다.

주요 흐름은 다음과 같다.

- `Transformer::createKVCachePlaceholders()`가 layer별 `cache_k_l<i>`,
  `cache_v_l<i>` placeholder를 생성한다.
- 각 모델의 attention 구성 코드가 `mha({q, k, v, cache_k, cache_v})` 형태로
  `mha_core`에 cache placeholder를 4번째, 5번째 입력으로 전달한다.
- `CausalLM::allocateAndBindKVCache()`가 `KVCacheManager`를 통해 layer별
  full cache tensor를 할당하고, compiled graph의 placeholder tensor에
  `setData()`로 해당 메모리를 연결한다.
- `MHACoreLayer::forwarding()`은 external cache mode에서 `context.getInput(3)`,
  `context.getInput(4)`로 cache tensor를 얻고, 내부에서
  `getSharedDataTensor()`로 write/read view를 만들어 attention을 수행한다.

이 구조는 external KV cache를 도입하는 데는 충분하지만, cache 양자화,
압축, eviction, paged attention 같은 최적화를 넣기에는 책임 분리가 약하다.
특히 현재 production path에서는 `KVCacheManager::getKeyCacheWriteView()` 같은
view helper가 사용되지 않고, `mha_core`가 cache tensor를 직접 다룬다.
따라서 optimizer를 `KVCacheManager` 옆에만 추가하면 실제 MHA 연산 경로에는
적용되지 않는다.

## 목표

1. KV cache의 물리 저장 포맷을 `mha_core`에서 분리한다.
2. raw FP32/FP16/UINT16 cache 경로를 유지하면서, INT8/Q4 등 quantized cache
   backend를 추가할 수 있게 한다.
3. cache position, save/load, layer/batch별 storage 관리를 단일 경로로 정리한다.
4. 초기 단계에서는 기존 MHA kernel을 최대한 재사용하고, 이후 fused quantized
   kernel로 확장할 수 있게 한다.
5. KV cache 최적화 설정은 `nntr_config.json`에서 읽어 모델 생성 시점에
   결정한다.
6. 성능 최적화 전에도 correctness와 bisect 가능성을 확보한다.

## 비목표

- 첫 단계에서 바로 Q4 fused attention kernel까지 구현하지 않는다.
- 기존 weight quantization용 `Q4_0_Tensor`, `Q6_K_Tensor`를 KV cache에
  그대로 재사용하지 않는다. 이 tensor들은 2D weight block quantization에
  가깝고, streaming append/read-window 방식의 KV cache와 제약이 다르다.
- 모든 모델에 대해 동시에 backend를 전환하지 않는다. raw backend를 먼저
  기본값으로 두고, opt-in 방식으로 quantized backend를 추가한다.

## 현재 구조의 제약

### `mha_core`가 cache layout을 알고 있다

`Applications/CausalLM/layers/mha_core.cpp`에서 cache write 위치와 read 범위를
직접 계산한다.

- write: `cache_index * cache_dim.width()` offset에 새 K/V를 기록한다.
- read: `[0, cache_to)` 범위의 shared tensor를 만들어 QK, AV 계산에 넘긴다.
- cache dtype은 `FP32` 또는 `UINT16/FP16`에 가깝게 분기되어 있다.

이 상태에서는 cache storage가 INT8, Q4, paged layout 등으로 바뀌면
`mha_core` 내부 offset 계산과 kernel 분기를 모두 수정해야 한다.

### position 상태가 중복되어 있다

- `KVCacheManager::cache_pos_`
- `MHACoreLayer::cache_index`

현재 `CausalLM::setKVCachePosition()`이 두 상태를 맞춰주고,
`mha_core`는 forwarding 후 자체적으로 `cache_index`를 증가시킨다.
하지만 optimizer가 append/evict/save/load를 관리하기 시작하면 position의
source of truth가 모호해진다.

### save/load가 raw tensor 저장에 묶여 있다

`KVCacheManager::save()`는 layer별 key/value tensor slice를 그대로 저장한다.
quantized cache에서는 다음 정보도 같이 저장해야 한다.

- cache format
- dtype 및 quantization scheme
- scale/zero-point metadata
- block size 또는 group size
- saved sequence length
- version/magic

## 제안 구조

`KVCacheManager`는 lifecycle과 graph binding을 담당하고,
`KVCacheOptimizer`는 물리 저장 포맷과 cache operation을 담당한다.

```cpp
enum class KVCacheFormat {
  RAW_FP32,
  RAW_FP16,
  RAW_UINT16_FP16_BITS,
  INT8_PER_TOKEN,
  INT8_PER_HEAD,
  Q4_BLOCK,
};

struct KVCacheSpec {
  unsigned int num_layers;
  unsigned int batch_size;
  unsigned int max_seq_len;
  unsigned int num_heads_kv;
  unsigned int head_dim;
  ml::train::TensorDim::Format tensor_format;
  ml::train::TensorDim::DataType compute_dtype;
  KVCacheFormat cache_format;
  std::string scale_granularity;
  ml::train::TensorDim::DataType materialize_dtype;
};

struct KVCacheRange {
  unsigned int layer;
  unsigned int batch;
  unsigned int from;
  unsigned int to;
};
```

```cpp
class KVCacheOptimizer {
public:
  virtual ~KVCacheOptimizer() = default;

  virtual void allocate(const KVCacheSpec &spec) = 0;
  virtual bool isAllocated() const = 0;

  virtual nntrainer::Tensor &getBindableKeyCache(unsigned int layer) = 0;
  virtual nntrainer::Tensor &getBindableValueCache(unsigned int layer) = 0;

  virtual void appendKey(const KVCacheRange &range,
                         nntrainer::Tensor &key_step,
                         bool apply_rope) = 0;
  virtual void appendValue(const KVCacheRange &range,
                           nntrainer::Tensor &value_step) = 0;

  virtual nntrainer::Tensor materializeKey(const KVCacheRange &range,
                                           nntrainer::Tensor &scratch) = 0;
  virtual nntrainer::Tensor materializeValue(const KVCacheRange &range,
                                             nntrainer::Tensor &scratch) = 0;

  virtual void save(std::ostream &out, unsigned int seq_len) const = 0;
  virtual void load(std::istream &in, unsigned int seq_len) = 0;
};
```

위 API는 최종 형태가 아니라 방향성이다. 특히 `appendKey()`에서 RoPE를
optimizer가 담당할지, `mha_core`가 RoPE 적용 후 optimizer에 넘길지는
구현 단계에서 결정해야 한다. 초기 단계에서는 기존 동작과의 차이를 줄이기
위해 `mha_core`가 RoPE 적용을 유지하고, optimizer는 storage 변환만 담당하는
것이 안전하다.

## Runtime 설정 계획

KV cache optimization 설정은 `nntr_config.json`에 둔다. 이 파일은 현재
`main.cpp`와 CausalLM API 경로에서 모델 생성 전에 로드되고, `Factory::create()`
를 통해 `Transformer` / `CausalLM` 생성자까지 전달된다. 또한
`Transformer::setupParameters()`가 이미 `batch_size`, `max_seq_len`,
`num_to_generate`, dtype 설정 등을 `nntr_cfg`에서 읽고 있으므로 KV cache
backend 선택도 같은 계층에서 처리하는 것이 자연스럽다.

권장 config schema:

```json
{
  "kv_cache": {
    "backend": "raw",
    "format": "raw_uint16_fp16_bits",
    "materialize_dtype": "fp16",
    "scale_granularity": "per_token_per_head",
    "fallback": "error"
  }
}
```

초기 기본값:

- `kv_cache` 섹션이 없으면 현재 플랫폼별 기본 raw cache 동작을 유지한다.
- `backend`: `raw`, `int8`, `q4` 같은 optimizer backend 선택값이다.
- `format`: backend 내부 storage format을 지정한다.
- `materialize_dtype`: materialize-to-scratch backend가 MHA kernel에 넘길
  scratch dtype이다.
- `scale_granularity`: quantized backend의 scale 단위다.
- `fallback`: unsupported 조합에서 `error`를 낼지, `raw`로 fallback할지
  결정한다.

예시:

```json
{
  "batch_size": 1,
  "max_seq_len": 4096,
  "num_to_generate": 256,
  "model_tensor_type": "FP32-FP32",
  "embedding_dtype": "Q6_K",
  "fc_layer_dtype": "Q4_0",
  "kv_cache": {
    "backend": "int8",
    "format": "int8_per_token",
    "scale_granularity": "per_token_per_head",
    "materialize_dtype": "fp16",
    "fallback": "error"
  }
}
```

구현 위치:

- `Transformer::setupParameters()` 또는 `CausalLM::setupParameters()`에서
  `nntr_cfg["kv_cache"]`를 parse한다.
- parsed result는 `KVCacheConfig` 같은 작은 struct에 저장한다.
- `CausalLM::allocateAndBindKVCache()`가 `KVCacheConfig`와 모델 파라미터를
  합쳐 `KVCacheSpec`을 만든다.
- `KVCacheManager::allocate(spec)`가 spec에 맞는 optimizer backend를 생성한다.
- CausalLM API의 hardcoded model config 경로도 같은 설정을 표현할 수 있어야
  한다. `api/model_config_internal.h`의 `ModelRuntimeConfig` 또는 이에 대응하는
  JSON 변환 경로에 `kv_cache_backend`, `kv_cache_format`,
  `kv_cache_materialize_dtype`, `kv_cache_scale_granularity`,
  `kv_cache_fallback` 필드를 추가하는 방식을 검토한다.

`generation_config.json`은 sampling/top-k/top-p/temperature 같은 decoding
policy에 남기고, KV cache optimization은 실행/runtime memory policy이므로
`nntr_config.json`에 둔다.

## 책임 분리

### `KVCacheManager`

- `KVCacheSpec` 생성
- backend 선택 및 소유
- cache position의 source of truth 관리
- model placeholder와 cache tensor binding
- save/load orchestration
- raw backend와 quantized backend 공통 API 제공

### `KVCacheOptimizer`

- 실제 cache storage 할당
- layer/batch/token별 append
- quantize/dequantize
- read range materialization
- scale/zero-point metadata 관리
- backend별 save/load payload 처리

### `MHACoreLayer`

- Q/K/V step tensor 처리
- RoPE, sink, causal/sliding-window semantics 유지
- optimizer/runtime API로 cache append/read 요청
- attention score 및 value aggregation 호출

장기적으로는 `MHACoreLayer`가 cache tensor의 물리 layout을 몰라야 한다.

## Backend 전략

### 1. Raw backend

`RawKVCacheOptimizer`는 현재 동작을 그대로 감싼다.

- storage: 현재와 동일한 `nntrainer::Tensor`
- format: `FP32`, `FP16`, `UINT16`
- append: 기존 `getSharedDataTensor()` + `copyData()` 또는 RoPE 결과 write
- materialize: full/read-range shared tensor 반환
- save/load: 기존 raw tensor 저장 방식 유지 또는 새 header format 도입 후 payload 저장

이 backend가 먼저 들어가야 이후 quantized backend와 비교할 기준이 생긴다.

### 2. INT8 backend

첫 quantized backend는 `Q4_0`보다 INT8을 권장한다.

이유:

- streaming append가 쉽다.
- token별 또는 head별 scale metadata가 단순하다.
- dequantize-to-scratch 구현이 작다.
- 정확도/속도 회귀를 분석하기 쉽다.

가능한 metadata layout:

```cpp
struct Int8KVCacheLayer {
  nntrainer::Tensor key_q;    // UINT8 or QINT8 storage
  nntrainer::Tensor value_q;
  nntrainer::Tensor key_scale;   // FP32, shape: B x 1 x T x H_KV
  nntrainer::Tensor value_scale; // FP32, shape: B x 1 x T x H_KV
  nntrainer::Tensor key_zero;    // optional
  nntrainer::Tensor value_zero;  // optional
};
```

초기 구현은 symmetric int8을 우선 고려한다.

- `q = clamp(round(x / scale), -127, 127)`
- `x = q * scale`
- scale granularity: per-token-per-head 우선

### 3. Q4 backend

Q4는 INT8 backend가 안정화된 뒤 추가한다.

주의할 점:

- `Q4_0_Tensor`는 현재 batch=1, channel=1, width divisible by 32 제약이 있다.
- KV cache는 `(batch, 1, max_seq_len, num_heads_kv * head_dim)`이고,
  batch/layer/token 단위 append가 핵심이다.
- 따라서 기존 weight quantized tensor를 그대로 쓰기보다 KV 전용 packed
  storage를 별도로 두는 편이 안전하다.

Q4 backend는 다음 두 방식 중 하나로 설계한다.

- token/head row 단위 packed block storage
- page 단위 packed block storage

## MHA 연산 통합 단계

### MVP: materialize-to-scratch

초기 quantized backend는 attention 계산 직전에 필요한 read range를 scratch
tensor로 dequantize한다.

흐름:

1. `mha_core`가 새 K/V step을 만든다.
2. optimizer가 step을 cache format으로 append한다.
3. optimizer가 `[0, to)` key/value range를 compute dtype scratch로 materialize한다.
4. 기존 `compute_kcaches()`와 `compute_fp16vcache_transposed()`를 그대로 호출한다.

장점:

- 현재 kernel 대부분을 재사용한다.
- correctness 검증이 쉽다.
- backend 추가와 kernel 최적화를 분리할 수 있다.

단점:

- dequantize 비용과 scratch 메모리 사용량이 있다.
- 긴 context에서는 bandwidth 이득이 일부 상쇄된다.

### Optimized: fused quantized kernel

MVP 이후 다음 kernel을 추가한다.

- `compute_kcaches_int8()`
- `compute_vcache_int8_transposed()`
- 추후 `compute_kcaches_q4()`, `compute_vcache_q4_transposed()`

이 단계에서는 materialize scratch 없이 quantized storage와 scale metadata를
직접 읽어 attention을 계산한다.

## API 변경 계획

### `KVCacheManager`

기존 public API는 가능하면 유지하되, 내부 storage 접근은 optimizer로 이동한다.

추가 후보:

```cpp
void allocate(const KVCacheSpec &spec);
void setOptimizer(std::unique_ptr<KVCacheOptimizer> optimizer);
KVCacheOptimizer &getOptimizer();
const KVCacheOptimizer &getOptimizer() const;

void setPosition(unsigned int pos);
void advance(unsigned int step_size);
unsigned int getPosition() const;

nntrainer::Tensor &getBindableKeyCache(unsigned int layer_idx);
nntrainer::Tensor &getBindableValueCache(unsigned int layer_idx);
```

기존 `getKeyCache()` / `getValueCache()`는 raw backend 호환을 위해 유지할 수
있지만, quantized backend에서는 의미가 달라진다. 따라서 장기적으로는
`getBindable*()`와 `getRaw*ForDebug()`처럼 의도를 분리하는 편이 좋다.

### `MHACoreLayer`

`mha_core`가 optimizer 객체에 직접 접근할 수 있는 방법이 필요하다.
선택지는 두 가지다.

#### 선택지 A: external tensor 유지 + side-channel registry

- 현재 placeholder binding 구조를 유지한다.
- `KVCacheManager`가 layer name 또는 cache tensor identity로 optimizer
  runtime을 registry에 등록한다.
- `mha_core`는 layer name으로 registry에서 runtime을 찾는다.

장점:

- graph input 구조 변경이 작다.
- 기존 model construction 영향이 적다.

단점:

- side-channel 의존성이 생긴다.
- lifecycle과 thread-safety를 신중히 관리해야 한다.

#### 선택지 B: cache runtime을 layer property/context로 명시 전달

- `MHACoreLayer`에 cache runtime pointer 또는 handle을 설정하는 API를 둔다.
- `CausalLM::allocateAndBindKVCache()`가 `forEachLayer()`로 각 MHA layer에
  layer별 optimizer handle을 주입한다.

장점:

- 의존성이 명확하다.
- `mha_core`가 cache tensor placeholder에 덜 의존한다.

단점:

- layer object에 application-level runtime pointer가 들어간다.
- serialization/export 시 제외해야 한다.

권장: B를 우선 검토한다. 현재도 `setKVCachePosition()`에서 `forEachLayer()`로
`MHACoreLayer::setCacheIndex()`를 호출하고 있으므로 유사한 패턴을 쓸 수 있다.

## 단계별 구현 계획

### Phase 0: 테스트 기준선 확보

- 현재 `unittest_kv_cache_manager` 실행 경로를 확인한다.
- raw cache save/load round-trip 테스트를 유지한다.
- MHA external cache mode에 대한 최소 regression test를 추가한다.
- prefill + decode 1 token 결과가 기존과 동일한지 확인할 수 있는 small model
  테스트를 준비한다.

완료 조건:

- raw backend 도입 전 현재 동작의 기준 결과가 있다.

### Phase 1: `KVCacheSpec`와 raw optimizer 도입

- `kv_cache_optimizer.h/.cpp` 추가
- `KVCacheOptimizer` abstract class 추가
- `RawKVCacheOptimizer` 구현
- `KVCacheConfig` parser 추가. `nntr_config.json`의 `kv_cache` 섹션이 없으면
  현재 raw 기본값을 만든다.
- `KVCacheManager` 내부 storage를 `RawKVCacheOptimizer`로 위임
- 기존 `getKeyCache()` / `getValueCache()` API는 raw backend에서 동일하게 동작

완료 조건:

- 기존 unit test가 raw optimizer 경유로 통과한다.
- 외부 observable behavior가 바뀌지 않는다.

### Phase 2: position source of truth 정리

- `KVCacheManager`를 position source of truth로 지정한다.
- `MHACoreLayer::cache_index`는 가능한 한 제거하거나, layer-local cached copy로만 둔다.
- `CausalLM::setKVCachePosition()`은 manager position 설정 후 MHA layer에
  runtime/position handle을 동기화한다.
- `advanceKVCachePosition()`이 실제로 사용되지 않는다면 제거 후보로 표시한다.

완료 조건:

- prefill, load-cache, decode에서 position mismatch 가능성이 줄어든다.
- non-zero cache load 후 다음 write 위치가 manager 기준으로 결정된다.

### Phase 3: MHA cache access adapter 추가

- `MHACoreLayer` 내부의 직접 cache view 생성 부분을 adapter 함수로 감싼다.
- raw backend에서는 기존 view와 동일한 tensor를 반환한다.
- 이 단계에서는 quantization 없이 구조만 바꾼다.

예시:

```cpp
auto cache_handle = getKVCacheRuntime();
cache_handle.append(layer_id, batch, from, to, key_step, value_step);
auto k_read = cache_handle.materializeKey(layer_id, batch, 0, to, scratch_k);
auto v_read = cache_handle.materializeValue(layer_id, batch, 0, to, scratch_v);
```

완료 조건:

- `mha_core`의 attention 계산부는 cache storage layout을 직접 가정하지 않는다.
- raw backend 결과가 기존과 동일하다.

### Phase 4: save/load format versioning

새 file format을 추가한다.

```text
magic: NNTR_KVCACHE
version: 1
format: RAW_FP16 / RAW_UINT16_FP16_BITS / INT8_PER_TOKEN / ...
num_layers
batch_size
max_seq_len
saved_seq_len
num_heads_kv
head_dim
payload_offset
metadata_offset
```

호환성 정책:

- 기존 raw-only cache 파일은 legacy loader로 읽을 수 있게 할지 결정한다.
- 새 format은 backend별 metadata를 반드시 저장한다.

완료 조건:

- raw backend save/load가 새 header로 round-trip 된다.
- invalid format/version에 대한 명확한 에러가 있다.

### Phase 5: INT8 materialize backend 구현

- `Int8KVCacheOptimizer` 추가
- per-token-per-head scale metadata 할당
- append 시 K/V step을 INT8로 quantize
- materialize 시 read range를 FP32 또는 FP16 scratch로 dequantize
- 기존 MHA kernel에 scratch tensor 전달

완료 조건:

- unit test에서 raw 대비 오차를 tolerance 내로 검증한다.
- prefill + decode smoke test가 통과한다.
- memory footprint가 raw 대비 감소하는 것을 로그 또는 test helper로 확인한다.

### Phase 6: `nntr_config.json` 기반 backend 선택 추가

KV cache optimization은 `nntr_config.json`의 `kv_cache` 섹션으로 선택한다.
CLI argument나 environment variable은 디버그 override로만 고려하고, 모델의
기본 runtime policy는 config 파일에 둔다.

예:

```json
{
  "kv_cache": {
    "backend": "int8",
    "format": "int8_per_token",
    "scale_granularity": "per_token_per_head",
    "materialize_dtype": "fp16",
    "fallback": "error"
  }
}
```

완료 조건:

- 기본값은 raw backend다.
- `nntr_config.json` opt-in으로 INT8 backend를 켤 수 있다.
- unsupported platform/backend 조합은 명확히 raw fallback 또는 error를 낸다.
- `nntr_quantize`가 새 `nntr_config.json`을 생성할 때 `kv_cache` 섹션을
  유지하거나 명시적으로 기본값을 기록한다.

### Phase 7: fused kernel 최적화

- INT8 K cache용 QK kernel 추가
- INT8 V cache용 AV kernel 추가
- scale metadata access pattern 최적화
- single-token decode path를 우선 최적화
- 이후 prefill/chunked path 확장

완료 조건:

- materialize backend 대비 latency 개선이 측정된다.
- raw backend 대비 memory 감소가 유지된다.

## 테스트 계획

### Unit tests

- `RawKVCacheOptimizer`
  - allocate dimensions
  - append/read range identity
  - multi-layer independence
  - batch offset correctness
  - save/load round-trip

- `Int8KVCacheOptimizer`
  - quantize/dequantize single token
  - per-token-per-head scale correctness
  - append multiple tokens
  - read partial range
  - saturation behavior
  - save/load metadata round-trip

- `KVCacheManager`
  - backend selection
  - position bounds
  - backend-independent save/load
  - invalid backend errors

### Integration tests

- small MHA external cache test
  - raw backend output equals current baseline
  - int8 backend output within tolerance

- CausalLM smoke test
  - prefill only
  - prefill + one decode token
  - load precomputed cache + decode

### Performance tests

- cache memory footprint
- prefill latency
- decode single-token latency
- long-context memory bandwidth sensitivity
- save/load file size

## 리스크와 대응

### 정확도 리스크

KV cache quantization은 attention logits에 직접 영향을 준다.

대응:

- INT8부터 시작한다.
- scale granularity를 per-token-per-head로 둔다.
- raw backend와 비교하는 tolerance test를 추가한다.
- layer별 error logging hook을 둔다.

### 성능 리스크

materialize-to-scratch 방식은 memory 절감은 되지만 latency가 악화될 수 있다.

대응:

- MVP 목적은 구조 검증으로 제한한다.
- decode single-token fused kernel을 다음 단계 목표로 둔다.
- scratch allocation을 매 step 하지 않고 재사용한다.

### 구조 리스크

optimizer runtime을 `mha_core`에 어떻게 전달할지가 가장 큰 설계 지점이다.

대응:

- raw backend adapter를 먼저 넣어 의존성 방향을 검증한다.
- side-channel registry보다 명시적 handle 주입을 우선 검토한다.
- export/serialization 대상에서 runtime pointer를 제외한다.

### 호환성 리스크

기존 precomputed cache 파일과 새 format이 충돌할 수 있다.

대응:

- magic/version header를 둔다.
- legacy raw loader 지원 여부를 별도 결정한다.
- format mismatch error message를 명확히 한다.

## 의사결정 필요 항목

1. `nntr_config.json`의 `kv_cache` schema 확정
   - `backend`와 `format`을 분리할지 여부
   - unsupported 조합의 기본 정책을 `error`로 할지 `raw` fallback으로 할지 여부

2. `mha_core`에 optimizer runtime을 전달하는 방식
   - layer handle 주입
   - registry lookup
   - external input tensor 확장

3. 초기 INT8 scale granularity
   - per-token
   - per-token-per-head
   - per-group

4. materialize dtype
   - FP32
   - FP16
   - platform별 선택

5. save/load 호환성
   - legacy raw cache loader 유지 여부
   - 새 format만 허용할지 여부

6. cache position ownership
   - manager 단일 source of truth
   - layer-local cache index 유지 여부

## 권장 순서 요약

1. raw backend를 `KVCacheOptimizer`로 감싸 현재 동작을 보존한다.
2. `mha_core`의 직접 cache 접근을 adapter/runtime 호출로 분리한다.
3. cache position ownership을 `KVCacheManager` 중심으로 정리한다.
4. save/load format에 version header를 추가한다.
5. `nntr_config.json`의 `kv_cache` 섹션으로 backend를 선택한다.
6. INT8 materialize backend를 opt-in으로 추가한다.
7. correctness와 memory footprint를 검증한다.
8. decode path부터 fused quantized kernel을 추가한다.

이 순서가 가장 안전한 이유는 storage abstraction, correctness, performance
optimization을 분리해서 진행할 수 있기 때문이다. 바로 Q4 또는 fused kernel로
들어가면 cache layout, position, save/load, kernel correctness 문제가 한 번에
겹쳐져 디버깅 비용이 커진다.
