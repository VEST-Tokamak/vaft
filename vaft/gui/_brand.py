import base64
import zlib

_Z = (
    "eNqVVMuO00AQ/JWW9wLSpDPvB4pz4MIJcUB74ZZ1HMdg7MjxbsLfU5PXPjgAkeLpaXWqq6rdWeyfGjr+7Pp9WWynafdhPj8cDn"
    "wwPIzNXEsp56goaFu3zXYqCysLOrTraXsOn9r68HE4lsUssZQ0s/mplH4+iuViXW/2y0XX9vVq/DSu1m3dT9SuywK4zeV+37cT"
    "KDzu6/HrblXVX/r7fV3QUZUF2vzCoYBFR32+n47lYj8NOxo2m309nfL5PquGbhjL4k7K8LDZFPM3ZeqPsrU9l81fU0TizHysq4"
    "muGtEckT1FFyMuQm8eXe8jfqMte1vQpu263OtBrXSdezWXVD/0dSY0Dj/qWe5frXZlMQ6P/fpV+vvQ9tf8crFbTVuCgZ914OiE"
    "12wcGc1RimBYJ7KGUxQhcNDkHCcngs813nKSwkfWlrxjH4TzrCIBQUthQdZTzjhhLGtH1rEMQid2gQzqkzCBlSUj2RlhA7pQ5p"
    "CES7nXSz7frgIw1rF7d9e8vym6GBdYWvnio7IzN20zBWYiQlUgxQESFGtDSLsoPKTFvzZQkvUrTMcKFlkOgDTsvIgnuzSeSiTJ"
    "NhKE+yiSzTKt5yBznCS5CIE5BocAXJfrgRMlhp1xgqIY2QEfXZDXbIOA4cETzA+w2marocIrgQEZSd6zswKDM4k8uiuh8epgBJ"
    "KNF8qxhdWRbRIqsNMX27UHAmmM5jQOixgcrLCRTSSVh5jHkTlgbsIn9qB/U/7/riV3s5ySztRCypSRv07on0Grdqy6mirsBl6f"
    "vCYVFsoZ0MPGIMnRP+/LeYVv2KdtyZva4Iv/peVvs3I3/A=="
)

LOGO_DATA_URI = "data:image/svg+xml;base64," + base64.b64encode(
    zlib.decompress(base64.b64decode(_Z))
).decode()

FAVICON_URL = LOGO_DATA_URI + "#favicon.svg"
