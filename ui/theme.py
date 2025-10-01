BACKGROUND_COLOR = "#050505"
SECONDARY_BACKGROUND_COLOR = "#12171A"
TEXT_COLOR = "#E5E9F0"
GRID_COLOR = "#1C2428"
ACCENT_COLOR = "#00E676"

# CSS to hide Streamlit Cloud toolbar and viewer/profile badges
CUSTOM_CSS = """
<style>
div[data-testid="stHorizontalBlock"] {
  display: flex;
  flex-direction: row;
}
@media (max-width: 640px) {
  div[data-testid="column"] {
    min-width: calc(30% - 1.5rem);
  }
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
