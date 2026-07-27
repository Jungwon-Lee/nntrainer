# Qwen3 CachedSlim MoE Optimization Review

## 1. 목적

이 문서는 Qwen3 CachedSlim MoE 추론 경로에서 추가로 최적화할 수 있는
부분을 정리한다. 주요 검토 대상은 다음과 같다.

- Decode 및 prefill 지연시간
- SIMD 활용
- 임시 tensor와 peak memory
- mmap 기반 expert weight I/O
- Expert cache hit rate
- Router 및 dispatch 비용

현재 구현에는 다음 최적화가 이미 적용되어 있다.

- Expert별 batched GEMM
- 실제 배정된 토큰 수만큼만 expert output을 할당하는 compact output
- Assignment 기반 output scatter-add
- PR #4154의 직렬 expert loop 유지
- Router, dispatch, mmap, GEMM, activation, reduction 단계별 프로파일링

이 문서의 예상 효과는 코드 구조를 기반으로 한 분석이며, 실제 효과는
대상 디바이스와 모델 weight 형식별 벤치마크로 확인해야 한다.

## 2. 우선순위 요약

| 우선순위 | 최적화 후보 | 주요 효과 | 구현 위험 |
|---|---|---|---|
| P0 | Router softmax 및 topK 통합 | Router 연산과 임시 할당 감소 | 낮음 |
| P0 | Small-K topK와 output buffer 재사용 | Decode router 할당 및 선택 비용 감소 | 낮음 |
| P0 | extra-topK stride 수정 | Expert cache 정책 정확성 개선 | 낮음 |
| P0 | Decode token input의 불필요한 할당 제거 | Decode allocator 비용 감소 | 낮음 |
| P0 | SIMD SwiGLU 사용 | Activation 연산 및 memory pass 감소 | 낮음 |
| P0 | Routing weight와 scatter-add 통합 | Expert output memory pass 감소 | 낮음 |
| P1 | Expert 결과 즉시 scatter 및 workspace 재사용 | Peak memory와 반복 할당 감소 | 중간 |
| P1 | 사용하지 않는 expert mask 제거 | Dispatch 시간과 scratch memory 감소 | 낮음 |
| P1 | Cache metadata 및 LRU 구조 개선 | Cache 관리 CPU 비용 감소 | 중간 |
| P1 | Cache 크기를 byte budget으로 설정 | Cache hit와 RSS 균형 개선 | 중간 |
| P2 | mmap prefetch pipeline | Flash I/O stall 감소 | 중간~높음 |
| P2 | Gate/Up multi-weight Q4_0 GEMV | Decode GEMV 및 thread overhead 감소 | 높음 |
| P2 | 여러 batch를 하나의 expert batch로 결합 | Batch 추론의 GEMM 효율 향상 | 중간 |

### 2.1 진행 현황

다음 항목은 `codex-qwen3-moe-batched-profile` 브랜치에 각각 독립된
커밋으로 적용되었다.

- [x] Router softmax 및 topK 통합 (`b2fe390f`)
- [x] extra-topK stride 수정 및 작은 expert 구성 지원 (`b2fe390f`)
- [x] Decode token input의 불필요한 할당 제거 (`012f1593`)
- [x] 단일 토큰 SIMD SwiGLU 적용 (`cf6c52cc`)
- [x] Routing weight와 scatter-add 통합 (`d5c91f90`)
- [x] CachedSlim의 사용하지 않는 expert mask 제거 (`df94b64b`)
- [x] 단일 compact expert-output workspace 및 즉시 scatter (`1d2df695`)
- [x] Gate/Up/activation workspace 재사용 (`65ff61d3`)
- [x] Expert assignment 연속 메모리화 (`2f844c18`)
- [x] Cache candidate deduplication (`93724b04`)
- [x] Dense-ID 기반 cache metadata 적용 (`71699d3a`)
- [x] Cache capacity 설정, cache counter 및 mmap prefetch pipeline
  (`6a091668`)
- [x] Batch-1 decode workspace의 memory-planner 재사용 (`44d4fdb5`)
- [x] ARM Q4_0 Gate/Up multi-weight GEMV (`a91496c7`)
- [x] Small-K router topK 및 candidate output 재사용 (`af4b99e2`)
- [ ] Allocation-free intrusive LRU 적용
- [ ] Cache capacity의 expert-byte budget 및 global budget 적용

아래 절에서 "현재"라고 표현한 코드는 최초 검토 시점의 구현을
의미한다. 위 진행 현황에서 완료로 표시된 항목은 작업 브랜치에서 이미
개선되었으며, 각 커밋을 독립적으로 비교하거나 되돌릴 수 있다.

## 3. Router softmax 및 topK 통합

현재 CachedSlim router는 다음 순서로 동작한다.

1. 모든 expert logits에 softmax를 적용한다.
2. `topK(topk + 5)`를 계산한다.
3. extra-topK 값을 정규화한다. 이 값은 이후 사용하지 않는다.
4. 동일한 router 결과에 `topK(topk)`를 다시 계산한다.
5. 선택된 topK 확률을 다시 정규화한다.

관련 코드:

- [`qwen_moe_layer_cached.cpp`](qwen_moe_layer_cached.cpp)
- [`FloatTensor::topK()`](../../../../nntrainer/tensor/float_tensor.cpp)

Softmax는 단조 함수이므로 raw logits의 순서와 softmax 결과의 순서는
동일하다. 또한 선택된 topK 확률을 다시 정규화하면 최종 weight는
다음과 같다.

```text
softmax(logit_i) / sum(softmax(selected_logits))
= exp(logit_i) / sum(exp(selected_logits))
```

따라서 다음 구조로 변경할 수 있다.

1. Raw logits에 `topK(topk + 5)`를 한 번만 실행한다.
2. 앞의 `topk` 항목을 실제 routing expert로 사용한다.
3. 선택된 `topk` logits에만 stable softmax를 적용한다.
4. 나머지 5개 index는 cache retention 후보로만 사용한다.

기대 효과:

- 전체 expert softmax 제거
- `topK()` 한 번 제거
- TopK output tensor와 내부 index vector 할당 감소
- 사용하지 않는 extra-topK value normalization 제거

현재 `FloatTensor::topK()`는 각 row마다 expert 수 크기의
`std::vector<size_t>`를 생성하고 `std::partial_sort()`를 수행한다.
따라서 동일 logits에 topK를 두 번 실행하는 비용은 피하는 것이 좋다.

작업 브랜치에서는 K가 16 이하일 때 stack 기반 worst-root heap을
사용한다. Qwen3의 기본 candidate 수인 `topk + 5`가 이 경로에
해당한다. 또한 candidate value/index tensor를 memory planner에
등록하여 batch-1 decode에서 `topK()` output을 매번 할당하지 않는다.
동일 logit은 낮은 expert index를 우선하므로 결과 순서가 결정적이다.

### 3.1 extra-topK stride 오류

`topK(topk + 5)`의 결과 row 크기는 `topk + 5`이지만 현재 index 접근은
다음 형태다.

```cpp
extra_indices_data[i * topk + k]
```

올바른 접근은 다음과 같다.

```cpp
extra_indices_data[i * (topk + 5) + k]
```

현재 방식은 두 번째 토큰부터 다른 위치의 index를 읽을 수 있다.
특히 prefill에서 잘못된 expert를 cache retention 후보로 사용할 수
있으므로 성능 최적화 전에 먼저 수정해야 한다.

## 4. Decode 경로의 불필요한 tensor 할당

현재 `compute_expert_forward()`는 `token_input`을 먼저 할당한다.
토큰 수가 하나이면 이 tensor를 input의 shared view로 다시 대입한다.

```cpp
Tensor token_input(token_input_dim);

if (num_tokens == 1) {
  token_input = input.getSharedDataTensor(...);
}
```

Decode에서는 대부분 expert당 토큰 수가 1이므로 최초 할당은 사용되지
않는다.

다음과 같이 변경할 수 있다.

```cpp
Tensor token_input;

if (num_tokens == 1) {
  token_input = input.getSharedDataTensor(...);
} else {
  token_input = Tensor(token_input_dim);
  // Gather assigned tokens.
}
```

기대 효과:

- 활성 expert마다 발생하는 hidden-size allocation 제거
- Decode allocator 호출 및 메모리 초기화 감소
- 단일 토큰 경로의 latency 편차 감소

## 5. SIMD SwiGLU 적용

현재 activation 경로는 토큰 수에 따라 다르게 동작한다.

- 단일 토큰: generic Swish 실행 후 별도 element-wise multiply
- 복수 토큰: SIMD `swiglu()` 호출

Qwen3 expert activation이 Swish/SwiGLU인 경우 단일 토큰에도
`nntrainer::swiglu()`를 직접 사용할 수 있다. nntrainer에는 ARM NEON과
x86 AVX2 구현이 이미 존재한다.

현재 경로:

```text
gate_out -> Swish -> acti_out
acti_out *= up_out
```

개선 경로:

```text
swiglu(acti_out, gate_out, up_out)
```

기대 효과:

- Activation과 multiply를 하나의 SIMD kernel로 결합
- Intermediate tensor memory pass 감소
- Decode의 generic activation callback 비용 감소

`MoEActivation`이 Swish 이외의 activation을 허용해야 한다면,
Swish일 때만 fused kernel을 사용하는 분기가 필요하다.

## 6. Routing weight와 scatter-add 통합

현재 compact expert output에 routing weight를 먼저 곱한 뒤 최종 output에
더한다.

```text
expert_output *= routing_weight
output += expert_output
```

이를 scaled add로 결합할 수 있다.

```text
output += routing_weight * expert_output
```

`Tensor::add_i(source, alpha)` 또는 backend의 SAXPY 계열 연산을 사용할
수 있다.

기대 효과:

- Expert output에 대한 별도 `sscal` pass 제거
- Weight 적용과 accumulation을 하나의 SIMD memory pass로 처리
- Standard, Slim, CachedSlim에 공통 적용 가능

수치 검증 시에는 기존 경로와 fused 경로의 FP32/Q4_0 허용 오차를 각각
확인해야 한다.

## 7. Expert 결과 즉시 scatter 및 workspace 재사용

PR #4154 이후 expert 외부 loop는 직렬로 실행된다. 따라서 모든 expert
output을 저장한 후 마지막에 한꺼번에 reduction할 필요가 없다.

현재 구조:

```text
expert 0 compute -> output 보관
expert 1 compute -> output 보관
...
모든 expert 계산 후 scatter-add
```

개선 구조:

```text
expert 0 compute -> 즉시 scatter-add
workspace 재사용
expert 1 compute -> 즉시 scatter-add
...
```

다음 tensor들은 expert loop가 직렬이므로 재사용할 수 있다.

- Gathered token input
- Gate output
- Up output
- Activation output
- Compact down-projection output

Prefill에서는 expert마다 배정 토큰 수가 다르므로 최대 assignment 수에
맞춘 workspace를 만들거나, 필요할 때만 확장하고 이후 재사용하는
방식이 적절하다.

Batch-1 decode의 Gate, Up, activation, down output은
`FORWARD_FUNC_LIFESPAN` tensor로 등록했다. 따라서 토큰마다 data buffer를
할당하지 않고 memory planner가 layer 실행 순서에 따라 재사용할 수
있다. Expert당 여러 토큰이 배정되는 prefill만 동적 크기 workspace로
fallback하므로 큰 prompt buffer가 decode 동안 유지되지 않는다.

기대 효과:

- `expert_outputs[num_experts]` 제거
- Expert output의 peak memory 감소
- Expert마다 반복되는 tensor allocation 감소
- Down projection 결과가 cache에 남아 있을 때 즉시 accumulation

주의 사항:

- Expert loop를 다시 병렬화하면 같은 토큰으로의 scatter가 data race를
  만들 수 있다.
- PR #4154의 직렬 expert loop를 유지하는 동안 적용하는 것이 안전하다.

## 8. Expert assignment를 연속 메모리로 구성

현재 assignment는 다음 형태다.

```cpp
std::vector<std::vector<std::pair<unsigned, float>>> expert_assignments;
```

이 구조는 호출마다 `num_experts`개의 vector 객체를 생성하고, 활성
expert마다 개별 heap allocation이 발생할 수 있다.

두 번의 선형 pass로 연속 메모리 형태를 만들 수 있다.

1. Expert별 assignment 수를 계산한다.
2. Prefix sum으로 expert별 offset을 만든다.
3. `total_tokens * topk` 크기의 연속 token-index/weight buffer를 채운다.

예상 구조:

```text
expert_offsets[num_experts + 1]
token_indices[total_routes]
router_weights[total_routes]
```

기대 효과:

- 많은 작은 heap allocation 제거
- Assignment 순회 및 gather의 cache locality 개선
- Active expert 목록 생성 단순화

Decode에서는 route 수가 작으므로 stack 또는 재사용 buffer도 검토할 수
있다.

## 9. 사용하지 않는 expert mask 제거

CachedSlim은 `expert_mask` tensor를 요청하지만 incremental inference에서
사용하지 않는다. Standard와 Slim도 mask를 기록하지만 expert
assignment는 topK indices에서 직접 생성한다.

Expert mask 크기는 다음과 같다.

```text
num_experts * topk * total_tokens * sizeof(float)
```

확인 후 제거할 대상:

- `expert_mask_idx`
- `requestTensor()` 호출
- `getTensor()` 및 `setZero()`
- TopK index를 mask에 기록하는 loop

기대 효과:

- Scratch tensor memory 감소
- Mask zero-initialization 제거
- TopK route 정보를 두 번 순회하는 비용 감소

Mask가 exporter, debug 또는 다른 실행 경로에서 사용되지 않는지 전체
검색과 모델 테스트로 확인해야 한다.

## 10. Expert cache metadata와 LRU 개선

최초 검토 시점의 cache는 다음 자료구조를 사용했다.

- `std::list<int>` 기반 LRU
- `std::unordered_map<int, list<int>::iterator>`
- `std::vector<bool> need_load`
- 고정 cache capacity 32

작업 브랜치에서는 candidate deduplication과 dense expert-ID metadata를
적용해 `unordered_map`, 사용되지 않는 prediction map, `vector<bool>`을
제거했다. `std::list<int>` 자체를 고정 배열 기반 intrusive LRU로
교체하는 작업은 아직 남아 있다.

Expert ID는 `0..num_experts-1` 범위의 조밀한 정수이므로 hash map 대신
index 기반 구조를 사용할 수 있다.

예시:

```text
resident[num_experts]
prev[num_experts]
next[num_experts]
lru_head
lru_tail
```

또는 expert 수가 작다면 timestamp 배열과 resident expert 선형 검색을
사용할 수 있다.

추가 개선:

- extra-topK 후보를 bitmap으로 deduplicate
- 동일 expert에 대한 반복 list erase/push 제거
- `unordered_map::find()` 후 `operator[]`로 다시 lookup하는 동작 제거
- 직렬 실행이 보장되는 범위에서는 cache mutex 필요성 재검토
- `vector<bool>` 대신 `vector<uint8_t>` 또는 별도 resident 상태 사용

Lock 제거는 layer instance가 여러 실행 context에서 동시에 호출될 수
있는지 확인한 후 결정해야 한다.

## 11. Cache capacity와 eviction 정책

기본 cache capacity는 expert 32개이며, 실행 시 다음 환경변수로 조절할
수 있다.

```bash
NNTR_MOE_CACHE_EXPERTS=48
```

값은 `num_experts` 이하로 clamp되며 0을 지정하면 invocation이 끝날 때
모든 expert mapping을 해제한다. 모델과 디바이스별 hit/RSS trade-off를
측정하면서 조절할 수 있지만, 아직 expert byte 크기나 여러 layer의
global budget을 자동으로 반영하지는 않는다.

남은 문제점:

- Expert 크기나 quantization dtype을 반영하지 않는다.
- 모든 MoE layer가 독립적으로 동일한 설정값을 사용한다.
- 새 expert를 activate한 후 eviction하므로 순간적인 peak가 증가한다.

후속 권장 방식:

- Expert 개수가 아니라 byte budget으로 결정
- 현재 invocation에서 사용할 expert는 eviction 대상에서 제외
- 새 expert를 activate하기 전에 불필요한 resident expert를 먼저 evict
- 여러 MoE layer를 포함하는 global memory budget 검토

Cache tuning에 필요한 profiler counter:

- Cache hit, miss, eviction: 적용 완료
- Expert별 reuse distance
- Activate/deactivate 시간
- Major/minor page fault
- Storage read bytes

`PROFILE` 빌드에서 `NNTR_MOE_PROFILE=N`을 설정하면 N번의 layer 호출마다
`cache_hits`, `cache_misses`, `cache_evictions`가 phase 시간과 함께
출력된다.

mmap이 유지된 상태와 실제 physical page resident 상태는 다르다.
따라서 mapping hit만으로 flash I/O가 없었다고 판단하면 안 된다.

## 12. mmap prefetch pipeline

최초 cache miss 경로는 다음 순서였다.

```text
Gate/Up/Down activate
즉시 expert compute
```

POSIX의 `Tensor::activate()`는 지원되는 플랫폼에서 `MADV_WILLNEED`를
사용하지만 activate 직후 weight를 읽으면 prefetch가 완료될 시간이
부족할 수 있다.

작업 브랜치에는 다음 두 단계 실행을 적용했다.

1. 현재 invocation에서 필요한 miss expert를 모두 확인한다.
2. Miss expert의 Gate/Up/Down을 먼저 activate하고 prefetch를 요청한다.
3. 이미 resident인 expert를 먼저 계산한다.
4. Prefetch를 요청한 expert를 계산한다.

기대 효과:

- mmap page fault와 GEMM의 overlap 가능
- Flash read stall 일부 은닉
- Down projection weight에 더 긴 prefetch 시간 제공

추가 후보:

- Gate/Up/Down file offset이 인접하면 expert 단위의 단일 mapping 검토
- Linux/Android에서 `madvise()` 또는 `posix_fadvise()` 정책 비교
- Cache miss 시 무조건 `WILLNEED`를 호출하는 것과 선택적 prefetch 비교

이 최적화는 storage와 OS page-cache 상태에 따라 결과가 크게 달라지므로
cold-cache와 warm-cache를 분리해 측정해야 한다.

## 13. Gate/Up multi-weight GEMV

Gate와 Up projection은 동일 input을 사용한다.

```text
token_input * gate_weight
token_input * up_weight
```

nntrainer의 여러 weight/output을 받는 `Tensor::dot()` 인터페이스를
CachedSlim Gate/Up에 연결했다. ARM CPU Q4_0의 batch-1 경로는 두
projection을 하나의 multi-weight GEMV로 전달한다.

적용된 Q4_0 multi-weight GEMV는 다음을 재사용한다.

- Input quantization 또는 packing
- Input cache line
- Thread dispatch
- Kernel setup

Prefill의 `M > 1`뿐 아니라 decode의 `M == 1`을 위한 batch-weight GEMV
kernel이 중요하다.

현재 구현 범위:

- ARM CPU의 Q4_0 batch-1 capability 추가
- Gate/Up 두 weight를 한 호출로 전달
- 기본 OMP backend를 weight/output-column chunk 단위로 병렬화
- 지원하지 않는 x86 및 다른 dtype은 기존 단일-weight fallback 유지

별도 GEMV 두 번과 multi-weight 결과가 동일한지 검증했으며, 실제
tokens/s 효과는 대상 ARM 디바이스에서 확인해야 한다. x86 전용
multi-weight kernel과 prefill용 CPU batch GEMM은 아직 후속 범위다.

## 14. Batch 차원 통합

현재 incremental forwarding은 input batch를 하나씩 순회한다. Batch가
1보다 크면 동일 expert에 배정된 서로 다른 batch의 토큰이 별도 GEMM으로
처리된다.

가능한 개선:

- 전체 `batch * sequence` 토큰을 한 번에 flatten
- Router와 assignment를 전체 batch에 대해 한 번 실행
- 동일 expert의 모든 batch token을 하나의 expert batch로 gather

기대 효과:

- Expert별 GEMM의 M 증가
- Expert activate/cache lookup 중복 감소
- Router/topK invocation 감소

일반적인 interactive decode는 batch 1이므로 효과가 제한적이지만,
server 또는 multi-request batching에서는 유효하다.

## 15. 프로파일러 자체 개선

현재 phase profiler는 여러 세부 구간에서 atomic duration을 누적한다.
프로파일 모드에서는 작은 phase의 측정값에 atomic 및 clock 호출 비용이
포함될 수 있다.

개선 후보:

- 한 invocation 동안 thread-local 또는 stack accumulator 사용
- `finish()`에서 한 번만 atomic/global counter에 반영
- Cache hit/miss/eviction counter 추가: 적용 완료
- Expert별 assignment 수 histogram 추가
- `num_tokens == 1`과 prefill 결과를 분리

프로파일이 비활성화된 일반 빌드에서는 현재와 같이 no-op을 유지해야
한다.

## 16. 권장 구현 순서

### 1단계: 낮은 위험의 application-level 개선

1. extra-topK stride 수정
2. Router topK 한 번으로 통합
3. 선택된 topK에만 stable softmax 적용
4. 사용하지 않는 extra-topK value normalization 제거
5. Decode token input의 불필요한 allocation 제거
6. 단일 토큰 SIMD SwiGLU 적용
7. Routing weight와 scatter-add 통합
8. 사용하지 않는 expert mask 제거
9. Cache hit/miss/eviction profiler counter 추가

### 2단계: memory와 cache 구조 개선

1. Expert 결과 즉시 scatter
2. Expert workspace 재사용
3. Assignment contiguous buffer 도입
4. Extra candidate deduplication
5. Dense-ID 기반 LRU 구조 적용
6. Cache capacity property 및 byte budget 도입

### 3단계: I/O와 backend 개선

1. mmap prefetch pipeline
2. Evict-before-activate 정책
3. Expert 단위 mapping 가능성 검토
4. Q4_0 Gate/Up multi-weight GEMV
5. Batch 차원 통합

## 17. 검증 계획

### 17.1 정확성

- 기존 구현과 최적화 구현의 FP32 output 비교
- Q4_0 output 허용 오차 비교
- Decode의 `num_tokens == 1` 검증
- Prefill의 여러 토큰이 같은 expert에 배정되는 경우 검증
- 여러 expert가 같은 token output에 누적되는 topK routing 검증
- Cache hit, miss, eviction을 강제로 발생시키는 작은 모델 테스트
- Batch가 1보다 큰 경우 검증

### 17.2 성능

다음 조건을 분리해서 측정한다.

| 구분 | 권장 값 |
|---|---|
| Prefill 길이 | 32, 128, 256, 512 |
| Decode | 128개 이상 토큰 연속 생성 |
| Thread 수 | 1, 4, 8 |
| Cache 상태 | cold, warm |
| Weight dtype | FP32, Q4_0, 필요 시 FP16 |
| Batch | 1, 지원 시 2 이상 |

수집할 지표:

- Prefill tokens/s
- Decode tokens/s
- Token latency p50/p95
- Router, dispatch, mmap, Gate/Up, activation, Down, reduce 시간
- Peak RSS/PSS
- Major/minor page fault
- Storage read bytes
- Expert cache hit/miss/eviction
- Expert별 assignment 수

### 17.3 적용 판단

- Router 시간이 크면 1단계 router 통합을 우선한다.
- Activation 시간이 크면 fused SIMD SwiGLU를 우선한다.
- Dispatch/reduce 시간이 크면 contiguous assignment와 scaled scatter를
  우선한다.
- mmap 또는 major fault가 크면 cache budget과 prefetch를 우선한다.
- Gate/Up/Down이 대부분이면 Q4_0 backend와 multi-weight GEMV를
  검토한다.
