# `vaft.machine_mapping` AttributeError 원인 정리

## 현상
- `import vaft` 는 성공하는데 `vaft.machine_mapping` 접근 시  
  `AttributeError: module 'vaft' has no attribute 'machine_mapping'` 발생.

## 원인 (가능성 높은 순)

### 1. **다른 위치의 vaft 패키지가 로드됨 (가장 유력)**

`import vaft` 시 Python이 **현재 repo가 아닌, 이미 설치된 다른 vaft**를 쓰는 경우입니다.

- 예: 예전에 `pip install vaft` 또는 `pip install -e /other/path/vaft` 로 설치한 패키지가 `sys.path` 앞에 있어, **repo의 `vaft`보다 먼저** 로드됨.
- 그 설치본의 `vaft/__init__.py` 에는 `from . import machine_mapping` 이 없거나, 예전 구조라서 `machine_mapping` 서브모듈이 없을 수 있음.
- 그러면 `import vaft` 는 성공하지만, `vaft` 에는 `machine_mapping` 이 붙지 않음.

### 2. **커널/캐시에 예전 vaft가 남아 있음**

- Jupyter/IPython 커널을 켠 **이후에** repo의 `vaft` 를 수정했을 수 있음.
- 이미 `import vaft` 가 한 번 실행된 상태면, `sys.modules['vaft']` 에 올라온 예전 버전이 그대로 유지됨.
- 그 예전 버전에는 `machine_mapping` 이 없을 수 있음.

### 3. **repo 기준 import 실패는 아님**

- 현재 repo의 `vaft/__init__.py` 는 다음 순서로 서브모듈을 불러옴:  
  `process` → `formula` → **`machine_mapping`** → `plot` → `omas` → `code` → `database`
- 여기서 `from . import machine_mapping` 이 **실패**하면, `import vaft` 자체가 예외를 일으키고, “성공했는데 속성만 없다”는 상황은 나오지 않음.
- 따라서 “import 는 되는데 `machine_mapping` 만 없다”면, **실제로 로드된 vaft는 이 repo의 최신 코드가 아닐 가능성이 큼**.

---

## 확인 방법

노트북이나 Python 세션에서 아래를 실행해 보세요.

```python
import vaft
print(vaft.__file__)
print(hasattr(vaft, 'machine_mapping'))
```

- **`vaft.__file__`**  
  - repo에서 로드됐다면: `/Users/yun/git/vaft/vaft/__init__.py` 같은 **repo 안 경로**가 나와야 함.  
  - `site-packages` / `dist-packages` 같은 **설치 경로**가 나오면, 그쪽(예전 설치본) vaft를 쓰는 것임.
- **`hasattr(vaft, 'machine_mapping')`**  
  - `False` 이면 위 1번 또는 2번 상황에 해당.

---

## 해결 방법

1. **이 repo를 개발 모드로 설치하고, 해당 환경에서 노트북 실행**
   - repo 루트에서:
     ```bash
     pip install -e .
     ```
   - 그 다음 **커널을 완전히 재시작**한 뒤 다시 `import vaft` 후 `vaft.__file__` 확인.

2. **Jupyter 커널 재시작**
   - “Kernel → Restart” 후, 위 확인 코드와 `vaft.machine_mapping` 사용을 다시 실행.

3. **노트북의 실행 경로 / `sys.path` 확인**
   - 노트북 첫 셀에:
     ```python
     import sys
     print(sys.path)
     ```
   - repo 루트(`/Users/yun/git/vaft`)가 `sys.path` 앞쪽에 오도록 실행 디렉터리나 `PYTHONPATH`를 맞추면, 같은 환경에서 repo 쪽 vaft가 우선 로드됩니다.

위 확인으로 “어느 vaft가 로드되는지”만 보면, `machine_mapping` 이 없는 이유를 거의 정확히 특정할 수 있습니다.
