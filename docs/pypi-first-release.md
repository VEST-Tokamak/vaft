# VAFT 최초 PyPI 릴리즈 안내

이 문서는 VAFT의 첫 PyPI 배포를 담당하는 사람을 위한 실행 절차입니다.
릴리즈 워크플로는 이미 저장소에 포함되어 있으며, PyPI API 토큰을 사용하지
않고 GitHub OIDC Trusted Publisher로 게시합니다.

## 배포 전 확인

1. 릴리즈 대상 커밋이 `develop` 또는 팀이 정한 보호 브랜치에 병합되어 있어야 합니다.
2. `vaft/version.py`의 `__version__`을 배포할 PEP 440 버전으로 변경합니다. 예: `0.1.0`.
3. README 또는 변경 기록에 주요 변경 사항을 정리합니다.
4. GitHub Actions의 **Package CI**가 통과했는지 확인합니다.

Package CI는 wheel과 sdist를 생성해 다음을 검사합니다.

- PyPI 허용 리소스만 포함되는지
- repository-only 샘플 및 sdist 테스트가 제외되는지
- wheel 크기가 25 MiB 이하인지
- 설치된 wheel에서 `import vaft`가 가능한지

## 1회 설정: PyPI Trusted Publisher

PyPI에서 VAFT 프로젝트를 열고 **Publishing** 설정에서 GitHub Trusted
Publisher를 추가합니다. 아직 프로젝트가 없으면 pending publisher로 먼저
등록할 수 있습니다.

입력값은 다음과 같습니다.

| 항목 | 값 |
| --- | --- |
| Owner | `VEST-Tokamak` |
| Repository | `vaft` |
| Workflow filename | `release-pypi.yml` |
| Environment | `pypi` |

GitHub 저장소에서도 **Settings → Environments**에 `pypi` Environment를 만들고,
가능하면 배포 전 maintainer 승인을 요구하도록 설정합니다. 이 설정 덕분에 태그가
push되어도 승인 전에는 PyPI 게시가 진행되지 않습니다.

PyPI API 토큰이나 `PYPI_TOKEN` secret은 만들거나 저장하지 마십시오. 게시 job은
`id-token: write` 권한으로 일회성 OIDC 자격 증명을 받습니다.

## 최초 릴리즈 실행

아래의 `X.Y.Z`는 `vaft/version.py`에 적은 버전과 정확히 같아야 합니다.

```bash
git checkout <release-commit>
git tag vX.Y.Z
git push origin vX.Y.Z
```

태그 push는 [Publish to PyPI](../.github/workflows/release-pypi.yml) 워크플로를
시작합니다. 이 워크플로는 다음 순서로 실행됩니다.

1. `vX.Y.Z` 태그와 `vaft.version.__version__`의 일치 여부 확인
2. wheel·sdist 생성
3. `twine check` 및 `scripts/verify_dist.py` 실행
4. 생성물을 artifact로 보관
5. `pypi` Environment 승인 대기
6. Trusted Publisher로 PyPI에 게시

Actions 탭에서 **Publish to PyPI** 실행을 열어 `pypi` Environment를 승인합니다.
게시가 완료되면 `https://pypi.org/project/vaft/X.Y.Z/`에서 버전과 파일을
확인합니다.

## 게시 실패 시

- 빌드 또는 배포물 검증 실패: 태그는 삭제하지 말고 원인을 수정한 새 커밋을 만든 뒤
  **새 버전**으로 다시 태그를 만듭니다.
- Trusted Publisher 인증 실패: Owner, Repository, workflow filename,
  Environment 이름이 위 표와 정확히 일치하는지 확인합니다.
- PyPI에 같은 버전이 이미 존재: PyPI 버전은 덮어쓸 수 없으므로 버전을 올려 새 태그를
  사용합니다.

## 이후 릴리즈

이후에는 버전 변경, Package CI 통과, `vX.Y.Z` 태그 push, `pypi` Environment 승인만
반복하면 됩니다. 상세 배포 정책은 [RELEASING.md](../RELEASING.md)를 참고하세요.
