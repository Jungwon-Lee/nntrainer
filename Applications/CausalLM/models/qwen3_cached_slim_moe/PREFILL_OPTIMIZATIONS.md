# Qwen3 Cached Slim MoE 긴 Context Prefill 최적화

이 문서는 Qwen3 Cached Slim MoE 모델에 긴 context가 입력될 때 prefill이
느려지는 원인과 이를 완화하기 위해 적용한 최적화를 설명합니다.

작업 기준은 `origin/main`의
`308bdd4f3b87a9dd5d5899104be32e6518efd637` 커밋입니다.

## 1. 먼저 알아두면 좋은 용어

- **Prefill**: 사용자가 입력한 prompt 전체를 한 번에 처리하여 KV cache를
  만드는 단계입니다. 입력이 길수록 한 번에 처리할 token 수가 많아집니다.
- **MoE(Mixture of Experts)**: 모든 token이 같은 FFN을 사용하는 대신,
  router가 token마다 일부 expert를 골라 계산하는 구조입니다.
- **Expert weight**: 각 expert가 사용하는 gate, up, down projection
  weight입니다.
- **Cached Slim**: 모든 expert weight를 항상 메모리에 두지 않고 필요할 때
  model file에서 가상 메모리로 매핑하며, 자주 쓰는 expert 일부만 cache에
  남기는 방식입니다.
- **Cache hit**: 필요한 expert가 이미 메모리에 매핑되어 있는 경우입니다.
- **Cache miss**: expert를 새로 `mmap`하고 storage page를 읽어야 하는
  경우입니다.
- **Resident expert**: 현재 프로세스 주소 공간에 매핑되어 사용 가능한
  expert입니다.

## 2. 긴 context에서 느려지는 이유

Qwen3 MoE는 token마다 소수의 expert만 선택합니다. 하지만 prompt가 길어지면
token 수가 많아지므로, prompt 전체로 보면 대부분의 expert가 한 번 이상
선택될 수 있습니다.

예를 들어 token마다 8개 expert 중 2개만 선택하더라도, 수천 개 token을
처리하는 동안에는 거의 모든 expert가 활성화될 가능성이 높습니다.

기존 구현은 한 번의 prefill에서 선택된 expert를 순서대로 활성화하면서
cache에 계속 추가한 뒤, 계산이 끝나는 시점에 cache 용량을 초과한 expert를
제거했습니다. 따라서 cache 용량이 32개인데 prompt가 100개 expert를
사용했다면, 계산 도중에는 최대 100개 expert가 동시에 매핑될 수 있었습니다.

이때 다음 비용이 함께 커집니다.

1. 많은 expert weight를 `mmap`하고 storage에서 page-in하는 I/O 비용
2. 사용 가능한 RAM을 넘을 때 발생하는 page reclaim과 page fault
3. expert별 중간 tensor를 반복해서 할당하고 해제하는 비용
4. 모든 expert의 출력 tensor를 계산 종료 시점까지 보관하는 메모리 비용
5. 실제 routing에 필요하지 않은 softmax와 추가 topK 연산

즉, token 하나당 활성 expert 수가 적다는 MoE의 장점이 긴 prompt 전체에서는
그대로 메모리 절감으로 이어지지 않았습니다.

## 3. 변경 전과 변경 후의 흐름

### 변경 전

```text
router 계산
  -> 전체 expert에 softmax
  -> topK와 추가 topK(topk + 5)를 각각 계산
  -> 활성 expert 0을 매핑하고 출력 보관
  -> 활성 expert 1을 매핑하고 출력 보관
  -> ...
  -> 활성 expert N을 매핑하고 출력 보관
  -> 모든 expert 출력 합산
  -> cache 용량을 넘은 expert 제거
```

긴 context에서는 활성 expert 수만큼 weight mapping과 출력 tensor가 동시에
쌓일 수 있습니다.

### 변경 후

```text
router logits에서 topK만 계산
  -> 선택된 logit에만 softmax
  -> 실제 routing 결과로 최종 32개 cache를 미리 계획
  -> 최종 cache에 포함되지 않는 기존 expert를 먼저 제거
  -> 현재 expert 계산
       + 다음 miss expert 하나만 미리 활성화
       + 현재 expert 출력 즉시 합산
       + 일시적으로 매핑한 expert 즉시 해제
  -> 계획한 expert만 cache에 유지
```

핵심은 **모든 활성 expert를 한꺼번에 메모리에 올리지 않는 것**입니다.

## 4. 적용한 최적화

각 최적화와 통합 브랜치에 적용된 커밋의 대응 관계는 다음과 같습니다.

| 최적화 | 커밋 |
|---|---|
| Expert cache 실제 상주 수 제한 | `4872b116` — `[CausalLM] Enforce Qwen3 CachedSlim expert cache limit` |
| 다음 cache miss 하나 prefetch | `81aa90e2` — `[CausalLM] Prefetch Qwen3 CachedSlim experts with a bounded window` |
| 실제 routing 기반 cache recency | `9ecc90b0` — `[CausalLM] Simplify Qwen3 CachedSlim expert routing` |
| 선택된 router logit만 softmax | `346e7fe1` — `[CausalLM] Normalize only selected Qwen3 MoE router logits` |
| Allocation-free small-k topK | `16b0cd11` — `[tensor] Add allocation-free small-k topK` |
| Expert scratch tensor 재사용 | `4c4c2e68` — `[CausalLM] Reuse Qwen3 MoE expert scratch buffers` |
| Expert output streaming | `1d05c16b` — `[CausalLM] Stream Qwen3 CachedSlim expert outputs` |
| Runtime page size 기반 mmap | `bff47bb7` — `[tensor] Support runtime page size for virtual tensor mmap` |

### 4.1 Expert cache의 실제 상주 수 제한

> 적용 커밋: `4872b116` —
> `[CausalLM] Enforce Qwen3 CachedSlim expert cache limit`

가장 영향이 큰 변경입니다.

계산을 시작하기 전에 이번 forward가 끝난 뒤 cache에 남아야 할 expert를
미리 결정합니다. cache 정책은 최근 routing 결과와 기존 LRU 순서를 이용하며,
최대 32개 expert만 선택합니다.

처리 순서는 다음과 같습니다.

1. 현재 cache와 이번 prompt의 활성 expert 목록을 확인합니다.
2. 최근 token에서 사용된 expert를 우선하는 최종 LRU cache를 계획합니다.
3. 최종 계획에 포함되지 않는 기존 cache expert를 먼저 해제합니다.
4. 최종 cache에 포함되는 miss expert는 활성화 후 cache에 유지합니다.
5. 최종 cache에 포함되지 않는 active expert는 계산할 때만 매핑합니다.
6. 일시적으로 매핑한 expert는 계산 직후 바로 해제합니다.

따라서 긴 prompt가 32개보다 많은 expert를 사용해도 steady cache에는 항상
최대 32개만 남습니다.

계산 중 최대 상주량은 다음과 같습니다.

```text
steady cache 32개
+ 현재 계산 중인 transient expert 최대 1개
+ 미리 활성화한 lookahead expert 최대 1개
```

즉, 최악의 경우에도 expert 수에 비례해 계속 증가하지 않고 대략
`cache 32 + current 1 + lookahead 1` 범위로 제한됩니다.

Expert weight는 gate, up, down projection 세 tensor가 하나의 묶음으로
활성화됩니다. `ExpertWeightLease`라는 RAII 객체가 이 세 tensor의 수명을
관리합니다. 일부 tensor만 활성화된 상태에서 예외가 발생하거나 expert 계산이
실패해도 이미 활성화한 weight를 자동으로 해제하여 mapping이 누적되지
않습니다.

### 4.2 다음 cache miss 하나만 prefetch

> 적용 커밋: `81aa90e2` —
> `[CausalLM] Prefetch Qwen3 CachedSlim experts with a bounded window`

현재 expert를 계산하는 동안 다음 cache miss expert 하나를 미리 활성화합니다.
설정 이름은 `moe_prefetch_distance`입니다.

```json
{
  "moe_prefetch_distance": 1
}
```

- `1`: 기본값입니다. 다음 miss expert 하나를 미리 활성화합니다.
- `0`: prefetch를 끄고 현재 expert를 계산할 때 활성화합니다.
- `2` 이상: 현재 구현에서는 허용하지 않으며 설정 오류를 발생시킵니다.

Linux와 Android에서는 virtual tensor 활성화 시 `MADV_WILLNEED`를 전달합니다.
운영체제는 다음 expert의 storage page를 미리 읽을 수 있으므로, 현재 expert의
GEMM/GEMV 계산과 다음 expert의 page-in이 일부 겹칠 수 있습니다.

prefetch 거리를 1로 제한한 이유는 다음과 같습니다.

- 여러 expert를 한꺼번에 prefetch하면 다시 메모리 사용량이 커집니다.
- 긴 context에서는 앞으로 사용할 expert가 많아 과도한 read-ahead가 발생할
  수 있습니다.
- 현재 expert 계산 시간 안에 다음 expert 하나를 준비하는 것만으로도 I/O
  stall을 줄일 가능성이 있습니다.

모든 platform에서 prefetch 효과가 같은 것은 아닙니다. `MADV_WILLNEED`를
지원하지 않거나 model file이 이미 OS page cache에 있다면 차이가 작을 수
있습니다. 그런 환경에서는 `moe_prefetch_distance=0`과 `1`을 각각 측정하는
것이 좋습니다.

### 4.3 Router 연산 축소

기존 router 경로에는 실제 expert 선택에 필요하지 않은 연산이 있었습니다.

#### 전체 softmax 제거

> 적용 커밋: `346e7fe1` —
> `[CausalLM] Normalize only selected Qwen3 MoE router logits`

기존에는 모든 expert logit에 softmax를 적용한 뒤 topK를 선택했습니다.
변경 후에는 raw logit에서 topK를 먼저 선택하고 선택된 값에만 softmax를
적용합니다.

Qwen3의 `norm_topk_prob` 동작에서는 다음 두 결과가 수학적으로 같습니다.

```text
전체 softmax -> topK -> 선택 확률 재정규화
raw logit topK -> 선택된 logit만 softmax
```

따라서 선택 결과와 routing weight는 유지하면서 선택되지 않은 expert에 대한
지수 함수 계산을 제거할 수 있습니다.

#### 두 번째 topK 제거

> 적용 커밋: `9ecc90b0` —
> `[CausalLM] Simplify Qwen3 CachedSlim expert routing`

기존 코드는 실제 routing을 위한 `topK(topk)` 외에 cache recency 계산을 위해
`topK(topk + 5)`를 한 번 더 수행했습니다. 이 결과는 실제 routing과 다를 수
있고, 값 tensor를 정규화하는 연산도 cache 정책에는 필요하지 않았습니다.

변경 후에는 실제 routing에서 나온 top-k index를 최근 token부터 역순으로
확인하여 cache recency를 직접 만듭니다. 이에 따라 다음 항목이 제거됩니다.

- 추가 `topK(topk + 5)`
- 추가 topK value 정규화
- `extra_top_k` 중복 저장
- 사용되지 않던 expert-mask tensor

#### Small-k topK 임시 할당 제거

> 적용 커밋: `16b0cd11` —
> `[tensor] Add allocation-free small-k topK`

MoE의 `topk`는 일반적으로 전체 expert 수보다 매우 작습니다. 이 경우를 위한
small-k 경로가 호출될 때마다 heap 메모리를 할당하지 않고, 이미 준비된 출력
tensor와 작은 고정 작업 공간을 사용하도록 변경했습니다.

동일한 값의 tie 처리, NaN 처리, 정렬 순서와 기존 출력 tensor 재사용도 단위
테스트로 확인합니다.

### 4.4 Expert 중간 scratch tensor 재사용

> 적용 커밋: `4c4c2e68` —
> `[CausalLM] Reuse Qwen3 MoE expert scratch buffers`

각 expert 계산에는 다음 중간 tensor가 필요합니다.

- 선택된 token을 모은 `token_input`
- gate projection 출력
- up projection 출력
- activation 출력

기존에는 expert를 처리할 때마다 이 tensor들을 새로 생성했습니다. 활성
expert가 많아질수록 큰 tensor 할당과 해제가 반복됐습니다.

변경 후에는 한 forward 안에서 가장 많은 token을 받은 expert의 크기를 먼저
계산하고, 그 크기에 맞춰 네 개의 scratch tensor를 한 번만 할당합니다.
각 expert는 필요한 크기만큼 shared view를 만들어 같은 backing memory를
재사용합니다.

```text
기존: expert 수만큼 중간 tensor 할당과 해제 반복
변경: forward당 네 개의 backing tensor를 한 번 할당하고 계속 재사용
```

이 변경은 storage I/O 자체를 줄이지는 않지만 allocator 비용과 peak memory
pressure를 줄여 page fault가 더 심해지는 것을 방지합니다.

### 4.5 Expert output을 하나씩 즉시 합산

> 적용 커밋: `1d05c16b` —
> `[CausalLM] Stream Qwen3 CachedSlim expert outputs`

기존에는 각 활성 expert의 compact output을 `expert_outputs` 배열에 저장하고,
모든 expert 계산이 끝난 뒤 최종 output에 합산했습니다.

긴 context에서 대부분의 expert가 활성화되면 전체 expert output이 동시에
메모리에 남습니다.

변경 후에는 다음 순서로 처리합니다.

1. expert 하나의 compact output만 생성합니다.
2. 해당 expert 계산을 수행합니다.
3. routing weight를 적용하면서 최종 token output에 즉시 scatter-add합니다.
4. compact output을 해제하고 다음 expert로 이동합니다.

따라서 expert output의 peak memory가 다음과 같이 바뀝니다.

```text
기존: 모든 활성 expert output 크기의 합
변경: 가장 큰 expert output 하나의 크기
```

expert를 오름차순으로 처리하고 기존과 같은 순서로 누적하므로 부동소수점
덧셈 순서도 유지합니다.

### 4.6 Virtual tensor의 실제 system page size 사용

> 적용 커밋: `bff47bb7` —
> `[tensor] Support runtime page size for virtual tensor mmap`

기존 virtual tensor mapping은 page 크기를 항상 4096 byte로 가정했습니다.
하지만 일부 Android, ARM, macOS 환경에서는 page 크기가 더 클 수 있습니다.
잘못된 offset으로 `mmap`하면 mapping이 실패하거나 expert 활성화 경로가
불안정해질 수 있습니다.

변경 후에는 `sysconf(_SC_PAGESIZE)`로 runtime page size를 확인하여
`mmap`과 `munmap` offset을 정렬합니다.

또한 `mmap`이 성공한 뒤에만 새 pointer를 tensor 상태에 반영하므로, 활성화
실패 시 이전 상태를 잘못 덮어쓰지 않습니다.

## 5. 어떤 변경이 속도에 가장 큰 영향을 주는가

긴 context에서 활성 expert 수가 cache 용량보다 많은 상황을 기준으로 예상
우선순위는 다음과 같습니다.

1. **Expert 상주 수 hard limit**
   - 모든 활성 expert를 동시에 매핑하던 문제를 직접 해결합니다.
   - RAM 부족, page reclaim, storage page-in이 병목일 때 가장 큰 효과가
     예상됩니다.
2. **다음 miss 하나 prefetch**
   - 남아 있는 동기식 page-in stall 일부를 현재 expert 계산과 겹칩니다.
   - cold cache와 느린 storage 환경에서 효과가 커질 수 있습니다.
3. **Expert output streaming과 scratch 재사용**
   - peak 임시 메모리와 allocator 부담을 줄입니다.
   - 긴 context에서 output과 intermediate tensor가 클수록 중요합니다.
4. **Router와 small-k topK 연산 축소**
   - 불필요한 CPU 연산과 임시 할당을 줄입니다.
   - expert weight I/O가 지배적인 환경에서는 전체 속도에 미치는 비율이
     상대적으로 작을 수 있습니다.
5. **Runtime page size 대응**
   - 직접적인 속도 개선보다 platform 안정성과 올바른 mapping을 위한
     변경입니다.

실제 순서는 장치 RAM, storage 속도, model precision, page cache 상태,
prompt의 routing 분포에 따라 달라질 수 있습니다.

## 6. 메모리 관점에서 보는 효과

긴 context에서 활성 expert 수를 `A`, cache 용량을 `C=32`라고 하면 expert
weight 상주 경향은 다음과 같습니다.

```text
기존 peak: 대략 A개 expert
변경 peak: 최대 C + current + lookahead
         = 대략 34개 expert
```

`A`가 32보다 작으면 hard limit의 효과는 작을 수 있습니다. 반대로 `A`가
전체 expert 수에 가까워질수록 차이가 커집니다.

여기서 expert 한 개는 gate, up, down projection weight를 모두 포함합니다.
현재 cache 제한은 byte가 아니라 expert 개수 기준이므로, model precision이나
expert dimension이 바뀌면 실제 메모리 사용량도 달라집니다.

## 7. 권장 성능 측정 방법

최적화 효과를 확인할 때는 decode token/s와 prefill 시간을 분리해야 합니다.
이 변경은 주로 긴 prompt의 prefill을 대상으로 합니다.

다음 조건을 기록하는 것이 좋습니다.

- prompt token 수
- 전체 expert 수와 token당 top-k
- prompt에서 실제로 선택된 unique expert 수
- cache hit와 miss 수
- model precision과 expert 하나의 weight 크기
- storage 종류와 읽기 대역폭
- process RSS와 peak RSS
- major/minor page fault 수
- `moe_prefetch_distance` 값

### Cold-cache 측정

model page가 OS page cache에 없는 상태를 가정합니다. 실제 첫 실행 latency와
storage I/O 효과를 보기 좋지만, 매 실행마다 동일한 cold-cache 조건을 만드는
방법은 platform별로 다릅니다.

### Warm-cache 측정

동일 prompt 또는 model을 반복 실행하여 model page가 이미 OS cache에 있는
상태를 측정합니다. 이 경우 router 연산, tensor 할당, output streaming의
효과가 상대적으로 더 잘 보입니다.

### 권장 비교 조합

```text
기존 origin/main
통합 브랜치 + moe_prefetch_distance=0
통합 브랜치 + moe_prefetch_distance=1
```

각 조합을 여러 번 실행하고 평균뿐 아니라 중앙값과 상위 latency도 함께
비교해야 page fault 변동을 구분하기 쉽습니다.

## 8. 검증한 항목

- runtime page size를 사용한 mapping offset 정렬
- virtual tensor activation 실패 시 상태 보존
- small-k topK의 일반 값, tie, NaN 및 출력 재사용
- 최근 token 우선 expert recency 수집
- 중복 expert 제거와 최종 LRU cache 계획
- 선택된 logit softmax와 기존 전체 softmax 기반 결과의 동등성
- Tiny Cached Slim MoE model 생성과 greedy generation
- 변경된 production 및 test translation unit의 `-Werror` syntax compile
- DEBUG 계측 경로 compile
- cache policy smoke test

전체 macOS native build는 현재 repository의 기존 platform 문제와 독립
worktree의 미초기화 subproject 때문에 완료하지 못했습니다. 확인된 환경
문제는 `malloc.h`, Mach VM type, 일부 ARM warning 및 OpenBLAS subproject
구성입니다.

## 9. 변경된 주요 파일

- `qwen_moe_layer_cached.cpp`
  - cache planning, expert lease, bounded prefetch, scratch 재사용,
    output streaming, router 경로 정리
- `qwen_moe_layer_cached.h`
  - prefetch property와 scratch 기반 expert forward interface
- `qwen_moe_cache_policy.h`
  - 최근 expert 수집과 최종 cache 계획
- `qwen3_cached_slim_moe_causallm.cpp`
  - `moe_prefetch_distance` 설정 전달
- `tensor.cpp`
  - runtime page size 기반 virtual tensor mapping
- `float_tensor.cpp`
  - allocation-free small-k topK
- 관련 unit test
  - cache policy, router normalization, topK, virtual mapping 검증

## 10. 현재 한계와 후속 작업

### Byte 단위 cache budget

현재 cache 용량은 32개 expert로 고정되어 있습니다. 그러나 FP32 expert와
quantized expert는 실제 byte 크기가 다릅니다. 향후에는 각 expert의 gate,
up, down tensor `getMemoryBytes()` 합을 이용한 byte budget이 필요합니다.

기존 expert-count 설정은 호환성을 위해 fallback으로 유지하는 것이 좋습니다.

### Gate/up weight fusion

Gate projection과 up projection을 fusion하면 일부 호출과 memory access를
줄일 수 있지만 serialized weight layout이 바뀝니다. 기존 model file과의
호환성을 고려해야 하므로 별도의 model-format migration으로 검토해야 합니다.

### 실제 장치 benchmark

현재 변경은 코드 경로와 단위 동작을 검증한 상태입니다. 최종 cache 크기와
prefetch 기본값은 목표 Android/Tizen 장치의 RAM, storage, page size를
기준으로 benchmark한 뒤 조정하는 것이 안전합니다.
