BACKGROUND_COLOR = "#050505"
SECONDARY_BACKGROUND_COLOR = "#12171A"
TEXT_COLOR = "#E5E9F0"
GRID_COLOR = "#1C2428"
ACCENT_COLOR = "#00E676"

# CSS to hide Streamlit Cloud toolbar and viewer/profile badges
HIDE_UI_CSS = """
<style>
div[data-testid="stToolbar"],
div[data-testid="stToolbarActions"],
.stToolbarActions,
button[title="View source"],
div[class*="viewerBadge"],
a[class*="viewerBadge"],
div[class*="profileContainer"],
div[class*="profilePreview"],
._container_gzau3_1,
._viewerBadge_nim44_23,
._profileContainer_gzau3_53,
._profilePreview_gzau3_63 {
  display: none !important;
}
</style>
"""

SIDEBAR_CSS = """
<style>
aside[data-testid=\"stSidebar\"],
div[data-testid=\"stSidebar\"] {
  width: 220px !important;
  min-width: 220px !important;
}
aside[data-testid=\"stSidebar\"] .block-container,
div[data-testid=\"stSidebar\"] .block-container {
  padding-top: 0.5rem;
}
</style>
"""
