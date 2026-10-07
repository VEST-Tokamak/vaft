<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-dark-1024.png">
    <img src="https://raw.githubusercontent.com/VEST-Tokamak/vaft/fc294b61bdb9e0d722f7eb1ea7c962ddda532959/docs/assets/brand/vaft-wordmark-1024.png" alt="VAFT" width="480">
  </picture>
</p>

# VAFT — 토카막을 위한 다목적 분석 프레임워크

[English](README.md) | 한국어 · [PyPI](https://pypi.org/project/vaft/) · [라이선스](LICENSE)

> **여러 분야의 핵융합 지식을 연결해 통합적인 토카막 연구를 돕습니다**

**VAFT는 장치에 종속되지 않는 토카막 연구를 위한 표준화되고 검증 가능하며 상호운용 가능한 과학 프레임워크입니다.** 실험 데이터, 재구성·시뮬레이션된 플라즈마 상태, 분석 과정을 공통 데이터 구조와 출처를 추적할 수 있는 결과로 연결합니다.

## VAFT가 연결하는 것

VAFT는 장치 고유의 측정값, [IMAS](https://imas.iter.org/)/[OMAS](https://gafusion.github.io/omas/) 표현, 데이터 처리·시각화, 분야별 물리 코드를 연결합니다. 표준 데이터는 원래의 과학 산출물을 보완하며, 연구자가 이를 비교하고 재사용할 공통 기반을 제공합니다.

![VAFT가 연결하는 핵융합 연구 생태계](https://raw.githubusercontent.com/VEST-Tokamak/vaft/develop/docs/assets/diagrams/fusion_research_ecosystem_presentation.svg)

[그림 설명과 상세 버전 보기](https://vest-tokamak.github.io/vaft/reference/diagrams/).

## 이를 가능하게 하는 네 관점

- **표현:** 장치 데이터와 플라즈마 상태를 출처·관례와 함께 상호운용 가능한 IMAS 구조로 옮깁니다.
- **연구 인프라:** 검증된 원본·표준 데이터, 노트북, 장치 지식을 찾고 공유합니다.
- **신뢰성:** 출처와 설정, 점검 결과를 기록해 추적·재현 가능한 과정에서 검증 가능한 결과를 만듭니다.
- **연구 방식과 이식성:** 실험, 재구성, 모델링, 해석을 하나의 연구 흐름으로 연결하고 다른 장치로 확장할 수 있게 합니다.

## VAFT로 할 수 있는 연구

[오프라인 예제](tutorial/README.md)로 시작해 [샷과 진단 데이터 탐색](https://vest-tokamak.github.io/vaft/workflows/data-access-imas/), [평형 재구성과 프로파일 피팅](https://vest-tokamak.github.io/vaft/workflows/equilibrium-kinetic-profiles/), [연구 노트북](notebooks/README.md)으로 이어갈 수 있습니다. 전체 흐름은 [워크플로 안내](https://vest-tokamak.github.io/vaft/workflows/start-here/)에 있습니다.

## VEST 참조 구현

서울대학교의 [VEST 토카막](https://vest-tokamak.github.io/vaft/reference/vest-systems/)은 장치 고유 데이터에서 표준 분석 결과까지 연결한 VAFT의 참조 구현입니다. VAFT의 데이터 모델과 워크플로 인터페이스는 VEST 밖의 연구에도 쓰이도록 설계되었습니다.

## 빠른 시작

공개된 패키지를 설치하고, 데이터베이스 계정이나 외부 물리 코드 없이 내장 예제를 살펴보세요.

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

- [문서 사이트](https://vest-tokamak.github.io/vaft/) · [데이터 접근](https://vest-tokamak.github.io/vaft/reference/database-data-sources/) · [평형 표현](https://vest-tokamak.github.io/vaft/reference/equilibrium-representations/)
- [튜토리얼](tutorial/README.md) · [노트북 목록](notebooks/README.md) · [기여 안내](CONTRIBUTING.md)
- [논문 인용과 감사의 글](https://vest-tokamak.github.io/vaft/reference/vest-tokamak-physics/) · [참고 자료](https://vest-tokamak.github.io/vaft/reference/references/) · [제3자 고지](THIRD_PARTY_NOTICES.ko.md)
