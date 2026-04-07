# vaft/data 데이터 카탈로그

`vaft/data` 내부 데이터 파일의 목적, 실제 구조(inspect 결과), 코드 종속 위치를 정리한다.
구조 정보는 파일을 직접 열어 확인했다(JSON key, MAT 변수/shape, HDF5 그룹, CSV 행/열, EFIT/AEQ 헤더).

## 1) JSON 샘플/테스트 데이터

| 파일 | 직접 inspect한 구조 | 주 용도 | 종속 스크립트/모듈 |
| --- | --- | --- | --- |
| `39915.json` | top-level keys: `coils_non_axisymmetric`, `dataset_description`, `em_coupling`, `equilibrium`, `magnetics`, `pf_active`, `pf_passive`, `spectrometer_uv`, `tf`, `wall`; `dataset_description.data_entry.pulse=39915`; `magnetics`에 `b_field_pol_probe/flux_loop/ip/time` 포함 | ODS 샘플(예제/시각화/테스트) | `vaft/omas/sample.py` (`sample_ods`, `sample_odc`), `test/contracts/test_contract_samples.py`, `notebooks/read_and_convert_data_structure.ipynb`, `notebooks/vest_daily_monitoring.ipynb`, `notebooks/vest_experimental_data_list.ipynb`, `notebooks/imas_omas_data_conversion.ipynb` |
| `41524.json` | `39915.json`과 동일 계열 구조; `dataset_description.data_entry.pulse=41524` | 멀티샷 ODC 샘플 | `vaft/omas/sample.py` (`sample_odc`), `notebooks/vest_daily_monitoring.ipynb` |
| `41672.json` | top-level keys: `barometry`, `dataset_description`, `em_coupling`, `equilibrium`, `magnetics`, `pf_active`, `pf_passive`, `spectrometer_uv`, `tf`; `dataset_description.data_entry`에 `pulse_type/run/user` 포함 | 샘플/계약 테스트 | `vaft/omas/sample.py` (`sample_odc`), `test/contracts/test_contract_samples.py`, `notebooks/vest_daily_monitoring.ipynb` |
| `thomson_scattering.json` | top-level key: `thomson_scattering`; 채널 수 5, time 길이 10 | Thomson IDS 계약 테스트 | `test/contracts/test_contract_samples.py`, `test/contracts/spec.py` |
| `vfit_ion_doppler_single.json` | top-level key: `charge_exchange`; 채널 수 1, time 길이 57, channel0 ion 2종 | Charge-exchange(single) 계약 테스트 | `test/contracts/test_contract_samples.py`, `test/contracts/spec.py` |
| `vfit_ion_doppler_profile.json` | top-level key: `charge_exchange`; 채널 수 70, time 길이 7, channel0 ion 1종 | Charge-exchange(profile) 계약 테스트 | `test/contracts/test_contract_samples.py`, `test/contracts/spec.py` |
| `omas_sample-checkpoint.json` | top-level keys: `barometry`, `dataset_description`, `em_coupling`, `magnetics`, `pf_active`, `pf_passive`, `spectrometer_uv`, `tf`, `wall`; `pulse=39033` | 대용량 체크포인트/실험 보관본 | 현재 코드 직접 참조 없음 |

## 2) 평형/재구성 파일

| 파일 | 직접 inspect한 구조 | 주 용도 | 종속 스크립트/모듈 |
| --- | --- | --- | --- |
| `g039915.00317` | 헤더: `EFIT ... # 39915 317ms ... 129 129` | gfile 샘플 로딩 | `vaft/omas/sample.py` (`sample_gfile`) |
| `g039915.00319` | 헤더: `EFIT ... # 39915 319ms ... 129 129` | 샤프라노프/프로파일 예제 | `test/test_shafranov_integral.py`, `notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb` |
| `g039020.031180` | 헤더: `VFIT ... # 39020 31180 ... 129 513` | SFL 변환 테스트 입력 | `test/test_sfl_conversion.py` |
| `a039915.00319` | AEQ 계열 텍스트; 상단에 날짜, shot `39915`, time `0.319000000E+03` 포함 | 샤프라노프 적분 레퍼런스 | `test/test_shafranov_integral.py` |

## 3) 진단 원천 데이터 (MAT/CSV)

| 파일 | 직접 inspect한 구조 | 주 용도 | 종속 스크립트/모듈 |
| --- | --- | --- | --- |
| `46051_NeTe.mat` | keys: `Ne`, `sigmaNe`, `Te`, `sigmaTe`, `time`; shape: `Ne/Te=(10,5)`, `time=(1,10)` | 신규 포맷 Thomson 원천 데이터 | `notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`, `workflow/automatic_pipeline_2_corrective_data_update/update_thomson_scattering_and_core_profile.py` (파일명 파싱 기반) |
| `NeTe_Shot39915_v9_rev.mat` | keys: `time_TS`, `poly1R1_* ... poly5R5_*`, `sigma`, `noise_level`, `minimum_chi2`; 주요 시계열 shape가 대부분 `(1,10)` | 레거시 Thomson 원천 데이터 | `notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb`, `workflow/automatic_pipeline_2_corrective_data_update/update_thomson_scattering_and_core_profile.py` (파일명 파싱 기반) |
| `CES_47514.mat` | keys: `LOS`, `LOS_err`, `temperature`, `temperature_err`, `velocity`, `velocity_err`; 전부 `(10,1)` 계열 | CES 예제 입력 | `vaft/machine_mapping/charge_exchange.py` (`charge_exchange`, `vfit_charge_exchange`) |
| `IDS_47518.mat` | keys: `Rposition(40,1)`, `time_IDS(1,4)`, `temperature/velocity/emissivity`와 error가 `(40,4)` | `charge_exchange(ids)` 또는 `charge_exchange(ion_doppler)` 변환 입력 | `vaft/machine_mapping/charge_exchange.py` (`read_doppler_ids_mat`), `notebooks/profile_fitting_using_equilibrium_and_kinetic_diagnostics.ipynb` |
| `digitizer_17592_45531.csv` | CSV shape: 40행 x 48828열 (각 행이 긴 파형 배열) | 원천 digitizer export | 현재 코드 직접 참조 없음 |
| `digitizer_22577_45531.csv` | CSV shape: 64행 x 48828열 | 원천 digitizer export | 현재 코드 직접 참조 없음 |

## 4) DB 필드 매핑

| 파일 | 직접 inspect한 구조 | 주 용도 | 종속 스크립트/모듈 |
| --- | --- | --- | --- |
| `sql_table.txt` | JSON dict 구조, 총 254개 엔트리(`signal name -> field_code`) | 신호명 기반 raw DB 로딩 | `vaft/database/raw.py` (`_sql_table_mapping`, `vest_load_by_name`), `vaft/machine_mapping/utils.py` (`load_raw_data`) |

## 5) HDF5/NetCDF 데이터 자산

| 파일 | 직접 inspect한 구조 | 주 용도 | 종속 스크립트/모듈 |
| --- | --- | --- | --- |
| `41514.h5` | HDF5 top-level: `/h5image` 단일 dataset (`{38245384}`) | 이미지/실험 자산 보관용으로 보임 | 현재 코드 직접 참조 없음 |
| `static_data_v1.h5` | HDF5 top-level: `/barometry`, `/pf_passive`, `/tf`; 하위에 IDS-style group/dataset 구조 | 정적 샘플 데이터 | 현재 코드 직접 참조 없음 |
| `testfile.h5` | `static_data_v1.h5`와 유사(`barometry/pf_passive/tf`) | 테스트/실험 샘플 | 현재 코드 직접 참조 없음 |
| `vest_imas_3.40.1.nc` | HDF5 컨테이너(NetCDF 확장자)로 `barometry`, `core_profiles`, `equilibrium`, `magnetics`, `thomson_scattering` 등 다수 IDS group 포함 | IMAS 구조 샘플/호환성 확인 | 현재 코드 직접 참조 없음 |

## 6) geometry/ 하위 자산

| 파일 | 직접 inspect한 구조 | 주 용도 | 종속 스크립트/모듈 |
| --- | --- | --- | --- |
| `geometry/MD.yaml` | keys: `source`, `description`, `channels`; `channels` 길이 75, 항목 키: `field_code`, `kind`, `calibration` | magnetics 채널 동적 보정 메타데이터 | `vaft/machine_mapping/magnetics.py` (`_load_md_channels`) |
| `geometry/VEST_MagneticsGeometry_Full_ver_2302.yaml` | keys: `source`, `description`, `channels`; `channels` 길이 75, 항목 키: `field_code`, `kind`, `r`, `z` | magnetics 정적 위치 메타데이터 | `vaft/machine_mapping/magnetics.py` (`_load_static_channels`), `vaft/machine_mapping/utils.py` (`_load_magnetics_channel_groups`) |
| `geometry/table.yaml` | keys: `source`, `description`, `entries`; `entries` 길이 75, 항목 키: `field_code`, `name` | field code 라벨 매핑 | `vaft/machine_mapping/magnetics.py` (`_load_names_by_code`) |
| `geometry/Coil_info.mat` | keys: `CoilCode`, `CoilGain`, `CoilNumber`; shape 모두 `(1,5)` | PF coil code/gain 매핑 | `vaft/machine_mapping/pf_active.py` (`vfit_pf`) |
| `geometry/VEST_DiscretizedCoilGeometry_Full_ver_1906.mat` | key: `DiscretizedCoilGeometry`, shape `(530,8)` | PF active static geometry (구버전 샷) | `vaft/machine_mapping/pf_active.py` (`vfit_pf_active_static`) |
| `geometry/VEST_DiscretizedCoilGeometry_Full_ver_2507.mat` | key: `DiscretizedCoilGeometry`, shape `(530,8)` | PF active static geometry (신버전 샷) | `vaft/machine_mapping/pf_active.py` (`vfit_pf_active_static`) |

## 7) Notes

- `vaft/machine_mapping/thomson_scattering.py`는 현재 다음 순서로 Thomson MAT를 자동 탐색한다: `thomson_scattering/NeTe_Shot{shot}_v9.mat` -> `thomson_scattering/NeTe_Shot{shot}_v9_rev.mat` -> `NeTe_Shot{shot}_v9.mat` -> `NeTe_Shot{shot}_v9_rev.mat` -> `{shot}_NeTe.mat`. 또한 explicit MAT 파일 경로 인자도 직접 수용한다.
- Thomson 동적 로더는 v9(`time_TS`, `poly*`)와 simple(`time`, `Te/Ne`, `sigmaTe/sigmaNe`) 스키마를 모두 지원한다.
- `digitizer_*.csv`, 일부 HDF5/NetCDF는 저장소 내부 직접 참조가 없어 현재는 수동 분석/아카이브 성격이 강하다.
