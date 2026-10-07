<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-dark-1024.png">
    <img src="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-1024.png" alt="VAFT" width="480">
  </picture>
</p>

# VAFT — 토카막을 위한 다목적 분석 프레임워크

[English](README.md) | 한국어 · [PyPI](https://pypi.org/project/vaft/) · [라이선스](LICENSE)

> **여러 분야의 핵융합 지식을 연결해 통합적인 토카막 연구를 돕습니다**

**VAFT는 IMAS 데이터 구조를 활용해 토카막 데이터를 정리하고 분석하는 과학 프레임워크입니다.** 실험 데이터, 재구성된 플라즈마 상태, 시뮬레이션 결과와 분석 과정을 공통 데이터 구조와 기록된 처리 이력으로 연결합니다.

## VAFT가 연결하는 것

VAFT는 장치별 진단·운전 데이터를 [IMAS Data Dictionary](https://imas-data-dictionary.readthedocs.io/en/latest/)가 정의한 공통 데이터 모델에 맞춰 옮깁니다. [OMAS](https://gafusion.github.io/omas/)는 이 구조를 다루는 Python 인터페이스입니다. VAFT는 표준 데이터로 처리·시각화 작업을 수행하고 EFIT·CHEASE 같은 기존 물리 코드와 데이터를 주고받습니다. 진단 원본 파일과 코드 출력도 표준 데이터와 함께 이용할 수 있습니다.

![VAFT가 연결하는 핵융합 연구 생태계](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/fusion_research_ecosystem_presentation.svg)

그림은 실험, 이론·모델링, 데이터 기반 연구가 측정·재구성·시뮬레이션된 플라즈마 상태를 함께 활용하는 모습을 보여 줍니다. VAFT는 연구자가 이 상태를 주고받고 비교하도록 돕되, 분야별 물리 코드를 대체하지는 않습니다.

[그림 설명과 상세 버전 보기](https://vest-tokamak.github.io/vaft/develop/reference/diagrams/).

## 이를 가능하게 하는 네 관점

![VAFT를 이루는 네 가지 역량](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/vaft_four_pillars.svg)

그림의 네 기둥은 차례로 수행하는 단계가 아니라 함께 작동하는 역량입니다. IMAS 매핑은 공통 인터페이스를 제공하고, 기록된 설정은 처리 과정을 추적할 수 있게 하며, 저장소와 아카이브는 결과를 해석하는 데 필요한 데이터와 장치 정보를 남깁니다. 연구자에게는 다음 네 가지가 중요합니다.

- **표현:** 진단 채널, 장치 형상, 평형과 프로파일을 출처·자속 관례와 함께 IMAS 구조에 맞춰 옮깁니다.
- **연구 인프라:** 진단 원본 파일과 샷별 표준 기록을 노트북·장치 구성 이력과 함께 찾아 쓸 수 있게 합니다.
- **신뢰성:** 교정값, 매핑 버전, 코드 설정과 품질 점검 결과를 기록해 작업 과정을 추적·재현하고 결과를 검증할 수 있게 합니다.
- **연구 방식과 이식성:** 재구성·모델링·시각화에 같은 데이터 경로를 사용하고, 다른 장치의 매핑을 마련해 워크플로를 확장합니다.

## 결과가 만들어지는 과정

![VAFT의 관리되는 과학 워크플로](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/scientific_workflow.svg)

장치 정보와 진단 측정값을 등록하고 IMAS 데이터 구조에 맞춰 옮깁니다. 진단 처리·평형 재구성·시뮬레이션은 이 구조를 읽고 기록하며, 워크플로는 설정과 출처를 남기고 데이터 품질을 점검해 결과를 분석에 쓸 수 있게 합니다.

## VAFT로 할 수 있는 연구

[오프라인 예제](tutorial/README.md)로 시작해 [샷과 진단 데이터를 탐색](https://vest-tokamak.github.io/vaft/workflows/data-access-imas/)하거나 [평형을 재구성하고 프로파일을 피팅](https://vest-tokamak.github.io/vaft/workflows/equilibrium-kinetic-profiles/)할 수 있습니다. [연구 노트북](notebooks/README.md)과 [워크플로 안내](https://vest-tokamak.github.io/vaft/workflows/start-here/)에 전체 예제가 있습니다.

## 여러 장치에 적용하는 구조

![VAFT의 장치 독립 구조](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/machine_agnostic_architecture.svg)

이 구조는 장치별 데이터 접근·매핑을 공통 IMAS 데이터 모델과 그 위의 분석 도구에서 분리합니다. 다른 장치를 연결하려면 해당 장치의 데이터 접근 방법과 매핑을 마련해야 하며, 모든 장치 지원이 구현된 것은 아닙니다.

## VEST 참조 구현

서울대학교의 [VEST 토카막](https://vest-tokamak.github.io/vaft/reference/vest-tokamak-physics/)은 이 구조를 실제로 적용한 참조 구현입니다. 그림은 VEST의 한 샷이 실험 데이터 처리를 거쳐 샷별 데이터베이스에 들어가는 흐름을 보여 줍니다.

![VEST 실험에서 분석까지 이어지는 데이터 플랫폼](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/vest_data_platform_overview.svg)

평형 재구성·물리량 추론과 시뮬레이션은 이 데이터베이스의 데이터를 읽고 결과를 다시 기록합니다. VAFT는 아래쪽에서 데이터 접근·분석을 담당하며, 각 단계의 구성 요소는 [VEST 플랫폼 상세 그림](https://vest-tokamak.github.io/vaft/develop/reference/diagrams/#the-vest-data-platform)에서 볼 수 있습니다.

## 빠른 시작

공개된 패키지를 설치하고, 데이터베이스 계정이나 외부 물리 코드 없이 패키지에 포함된 VEST 예제를 살펴보세요.

```bash
pip install vaft
```

```python
import vaft

ods = vaft.omas.sample_ods()
print(sorted(ods.keys()))
```

그림을 그리는 첫 예제는 [시작 안내](https://vest-tokamak.github.io/vaft/workflows/start-here/)에 있습니다. 소스 설치와 운영체제별 환경 설정은 [install/README.md](install/README.md)를 참고하세요.

## 자세한 문서

- [문서 사이트](https://vest-tokamak.github.io/vaft/) · [IMAS 개념](https://vest-tokamak.github.io/vaft/reference/imas-concepts/) · [데이터 접근](https://vest-tokamak.github.io/vaft/reference/database-data-sources/) · [평형 표현](https://vest-tokamak.github.io/vaft/develop/reference/equilibrium-representations/)
- [튜토리얼](tutorial/README.md) · [노트북 목록](notebooks/README.md) · [기여 안내](CONTRIBUTING.md)
- [논문 인용과 감사의 글](https://vest-tokamak.github.io/vaft/reference/vest-tokamak-physics/) · [참고 자료](https://vest-tokamak.github.io/vaft/reference/references/) · [제3자 고지](THIRD_PARTY_NOTICES.ko.md)
