# VAFT - 토카막을 위한 다목적 분석 프레임워크

<!-- README.ko.md -->
[English](README.md) | 한국어

[PyPI](https://pypi.org/project/vaft/)
[Python](https://pypi.org/project/vaft/)
[License](LICENSE)

**VAFT**는 서울대학교의 [VEST (Versatile Experiment Spherical Torus)](https://eng.snu.ac.kr/) 토카막을 위한 전용 데이터 플랫폼이면서, IMAS 데이터 모델을 기반으로 한 장치 및 코드 범용 데이터 분석 프레임워크인 오픈 소스 Python 라이브러리입니다. [OMAS](https://gafusion.github.io/omas/) 인터페이스 라이브러리와 [HSDS](https://github.com/HDFGroup/hsds) 원격 HDF5 데이터베이스를 기반으로 [IMAS](https://imas.iter.org/) 호환 데이터 인터페이스를 제공합니다.

> Hong-Sik Yun, Sunjae Lee *et al* 2025 *Plasma Phys. Control. Fusion* **67** 115021
> ([doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a))

## 주요 기능

| 기능 | 설명 |
| --- | --- |
| **원격 데이터베이스 접근** | 단일 함수 호출로 VEST HSDS 서버에서 샷별 OMAS ODS 데이터를 불러옵니다. |
| **장치 매핑** | VEST 고유 진단 신호를 표준 IMAS IDS로 변환합니다(자기 진단, 톰슨 산란, 바로미터, PF active, TF, UV 분광기, 전하 교환 등). |
| **평형 및 안정성** | EFIT, CHEASE, GPEC(DCON/RDCON) 인터페이스를 제공하며 IDS 형식의 코드 입출력을 지원합니다. |
| **물리 공식** | 평형 물리량(폴로이달/토로이달 자속, 안전 계수), 안정성 지표(베타 한계, 풍선 모드), 가둠 시간 스케일링 법칙(ITER89P, H98y2), Green 함수를 제공합니다. |
| **신호 처리** | 평활화, 기준선 제거, 잡음 저감, 전자기장 계산, 와전류 모델링을 지원합니다. |
| **프로파일 피팅** | 운동론 진단(톰슨 산란, CES)을 평형 자속면에 매핑하고 GP, 다항식 또는 지수 모델로 피팅합니다. |
| **시각화** | 시간 파형, 1D/2D 프로파일, 자속면 등고선, 상면도, 운전 공간 지도를 제공합니다. |
| **IMAS 상호운용성** | OMAS ODS와 IMAS-Python(AL5) 데이터 구조 간 변환 및 NetCDF 내보내기를 지원합니다. |

## 아키텍처

```
VEST 데이터 분석 플랫폼
├── 자동화 파이프라인(Snakemake)       ── 실험 → 후처리 → 시뮬레이션
├── 데이터베이스(IMAS-HSDS)             ── REST API를 통한 샷별 HDF5 저장소
└── 인터페이스(VAFT)                    ── 데이터 접근, 매핑, 처리, 시각화
```

### VEST 데이터베이스에서 사용할 수 있는 IMAS IDS

**실험 데이터:**
`dataset_description` · `magnetics` · `tf` · `pf_active` · `barometry` · `spectrometer_uv` · `thomson_scattering` · `charge_exchange`

**모델링 데이터:**
`wall` · `em_coupling` · `pf_passive` · `equilibrium` (EFIT/CHEASE) · `core_profiles` · `mhd_linear` (DCON/RDCON)

## 빠른 시작

### 설치

소스에서 설치(권장):

```bash
git clone https://github.com/VEST-Tokamak/vaft.git
cd vaft
python -m pip install -e .
```

```bash
# 개발 도구
python -m pip install -e ".[dev]"
```

#### 레거시 NumPy 1 설치

NumPy 1을 요구하는 외부 패키지가 있을 때에만 사용하세요. `h5pyd==0.20.0`이 NumPy 2를 요구한다고 선언하는 이슈가 있으므로, NumPy를 교체한 뒤 `h5pyd`는 `--no-deps`로 설치합니다.

```bash
python -m pip install -e .
python -m pip install --force-reinstall --no-deps "numpy>=1.26.4,<2"
python -m pip install --force-reinstall --no-deps h5pyd==0.20.0
```

이는 레거시 호환성 옵션이며, `pip check`는 의도적으로 우회한 NumPy 요구 사항을 보고할 수 있습니다.

PyPI에서 설치(더 이상 권장하지 않음):

```bash
pip install vaft
```

**지원 Python 버전**: 3.10 -- 3.13
**기본 수치 연산 스택**: NumPy 2.x (`numpy>=2.0.0,<3`)

외부 코드 설치 루트와 VAFT 런타임 경로는 프로세스 환경 변수로 설정합니다.
`{CODE}HOME` 디렉터리 구조, 호환 변수 및 셸 설정 예시는
[외부 핵융합 코드 초기화 노트북](notebooks/initialize_external_fusion_codes.ipynb)을 참고하세요.

### VEST 데이터베이스 연결

원격 VEST HSDS 데이터베이스를 사용하려면 HSDS 자격 증명을 설정하세요.

```bash
hsconfigure
```

프롬프트에 다음 값을 입력합니다.

| 항목 | 값 |
| --- | --- |
| 서버 엔드포인트 | `http://147.46.36.244:5101` |
| 사용자 이름 | [peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr)에 문의 |
| 비밀번호 | [peppertonic18@snu.ac.kr](mailto:peppertonic18@snu.ac.kr)에 문의 |

`connection ok` 메시지가 표시되면 연결된 것입니다. 자세한 내용은 [상세 안내서](https://vest-tokamak.github.io/vaft/guide/Quick_start_guide/)를 참조하세요.

### 기본 사용법

```python
import vaft

# 원격 데이터베이스에서 샷 불러오기
ods = vaft.database.load(39915)

# IMAS 구조 데이터를 직접 접근
time = ods['magnetics.time']
ip = ods['magnetics.ip.0.data']
```

### 프로파일 피팅

```python
# 톰슨 산란 데이터를 평형 자속 좌표에 매핑한 뒤 프로파일 피팅
mapped_rho = vaft.process.equilibrium_mapping_thomson_scattering(ods, geq)
vaft.process.profile_fitting_thomson_scattering(
    ods, time_ms, mapped_rho, fitting_function_te='gp', fitting_function_ne='gp'
)
```

### IMAS 변환

```python
# OMAS ODS ↔ IMAS-Python 데이터 엔트리 변환
from vaft.imas import omas_imas
omas_imas.save_omas_imas(ods, pulse=39915, run=0)
```

## 라이브러리 모듈

```
vaft/
├── database/          # 원격 데이터베이스 접근(HSDS, raw SQL)
├── machine_mapping/   # 장치 고유 진단 신호를 IDS로 변환(70개 이상 함수)
├── formula/           # 물리 공식(평형, 안정성, Green 함수)
├── process/           # 신호 처리, EM 모델링, 프로파일 피팅
├── plot/              # 시각화(시간, 1D, 2D, 상면도, 분석)
├── omas/              # ODS 유틸리티(샷 메타데이터, 샘플 데이터)
├── imas/              # IMAS-Python(AL5) 상호운용성
├── code/              # 코드 인터페이스(EFIT, CHEASE, GPEC, TES, TokaMaker, Snakemake)
└── data/              # 샘플 데이터, 형상 자산, 보정 테이블
```

## 예제 노트북

| 노트북 | 설명 |
| --- | --- |
| [database_initialization_and_load](notebooks/database_initialization_and_load.ipynb) | 핵심 데이터 로딩 및 프레임워크 기초 |
| [plotting_sample_using_vaft_plot_module](notebooks/plotting_sample_using_vaft_plot_module.ipynb) | plot 모듈을 사용한 시각화 예제 |
| [profile_fitting_using_equilibrium_and_kinetic_diagnostics](notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb) | 톰슨/CES 매핑 및 프로파일 피팅 |
| [read_and_convert_data_structure](notebooks/read_and_convert_data_structure.ipynb) | ODS/IMAS 데이터 구조 변환 |
| [imas_omas_data_conversion](notebooks/imas_omas_data_conversion.ipynb) | IMAS ↔ OMAS 상호운용성 |
| [vest_experimental_data_list](notebooks/vest_experimental_data_list.ipynb) | VEST 샷 데이터베이스 탐색 |
| [confinement_time_scaling](notebooks/confinement_time_scaling.ipynb) | 에너지 가둠 시간 스케일링 분석 |
| [vest_daily_monitoring](notebooks/vest_daily_monitoring.ipynb) | 일일 실험 모니터링 대시보드 |
| [publication_figures](notebooks/publication_figures.ipynb) | 출판물 그림 재현 |
| [verify_exist_shot_and_load](notebooks/verify_exist_shot_and_load.ipynb) | 샷 존재 여부 확인 및 TS/CX 데이터 불러오기 |
| [tokamak_power_balance](notebooks/tokamak_power_balance.ipynb) | 토카막 전력 수지 및 복사 성분 분해 |
| [verification_and_validation](notebooks/verification_and_validation.ipynb) | 검증 및 유효성 확인 예제 |
| [soft_x_ray_signal_analysis](notebooks/soft_x_ray_signal_analysis.ipynb) | 연 X선 신호 분석 |
| [equilibrium_refinement_using_chease](notebooks/equilibrium_refinement_using_chease.ipynb) | CHEASE를 이용한 평형 정교화 |
| [forward_equilibrium_using_TES](notebooks/forward_equilibrium_using_TES.ipynb) | TES를 이용한 순방향 평형 재구성 |
| [forward_equilibrium_using_TokaMaker](notebooks/forward_equilibrium_using_TokaMaker.ipynb) | TokaMaker(Open FUSION Toolkit)를 이용한 순방향 자유경계 평형 계산 |
| [time_dependent_equilibrium_using_TokaMaker](notebooks/time_dependent_equilibrium_using_TokaMaker.ipynb) | TokaMaker를 이용한 진공용기 와전류·벽 고유모드·준정적 시간 전개 |
| [kinetic_efit_end_to_end](notebooks/kinetic_efit_end_to_end.ipynb) | 엔드투엔드 kinetic-EFIT 워크플로 |

## 관련 자료

- **문서**: [vest-tokamak.github.io/vaft](https://vest-tokamak.github.io/vaft/)
- **논문**: H.-S. Yun, S. Lee *et al*, "Developing an IMAS-compatible platform for the university-scale tokamak VEST and its application to operating characteristics analysis", *Plasma Phys. Control. Fusion* **67** 115021 (2025). [doi:10.1088/1361-6587/ae1b6a](https://doi.org/10.1088/1361-6587/ae1b6a)
- **OMAS**: [gafusion.github.io/omas](https://gafusion.github.io/omas/) — IMAS 데이터 구조를 위한 Python API
- **OMFIT**: [omfit.io](https://omfit.io/) — 통합 모델링 및 실험 데이터 분석 프레임워크
- **HSDS**: [github.com/HDFGroup/hsds](https://github.com/HDFGroup/hsds) — HDF5 REST 기반 데이터 서비스
- **IMAS**: [github.com/iterorganization/IMAS-Data-Dictionary](https://github.com/iterorganization/IMAS-Data-Dictionary) — ITER 통합 모델링 및 분석 도구 모음

## 기여

기여를 환영합니다. [이슈](https://github.com/VEST-Tokamak/vaft/issues)를 열거나 풀 리퀘스트를 제출해 주세요.

데이터베이스 쓰기 권한은 [peppertonic18@snu.ac.kr, satelite2517@snu.ac.kr](mailto:peppertonic18@snu.ac.kr)로 문의하세요.

## 감사의 글

저자들은 기술적 조언을 제공한 General Atomics의 O Meneghini와 J McClenaghan에게 감사드립니다. 데이터 처리의 일부는 OMFIT 통합 모델링 프레임워크의 코드 API를 사용하여 수행되었습니다[1]. 이 연구는 한국 정부(MSIT)가 지원하는 한국연구재단(NRF) 연구비(RS-2021-NR057187, RS-2023-00281276, RS-2024-00409564, RS-2025-02304810)의 지원을 받았습니다.

## 제3자 고지

### OPEN-ADAS 원자 과정 루틴

VAFT의 OPEN-ADAS ADF11 파싱, 보간, 기본 파일 선택 및 이온화 평형 로직 일부는 아래 라이선스로 배포되는 소프트웨어를 수정하거나 이를 기반으로 작성되었습니다.

> MIT License
>
> Copyright (c) 2021 Francesco Sciortino
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### OMFIT classes 호환성 포트

VAFT의 네이티브 EQDSK 호환성 및 상호운용 경로에는 `omfit_classes`에서 이식하거나 수정한 동작이 포함되어 있습니다. VAFT는 해당 레거시 NumPy, SciPy 및 xarray 인터페이스를 위한 호환성 shim도 제공합니다. 원본 OMFIT classes 소프트웨어는 아래 라이선스로 배포됩니다.

> Copyright 2013-2021 the OMFIT contributors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
