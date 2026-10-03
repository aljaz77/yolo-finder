from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

ACCENT = "#3d8bfd"

STYLE = f"""
QWidget {{ font-size: 10pt; }}
QListWidget#nav {{ background: #1b1d22; border: none; padding-top: 8px; font-size: 11pt; }}
QListWidget#nav::item {{ padding: 10px 14px; border-left: 3px solid transparent; color: #c8ccd4; }}
QListWidget#nav::item:selected {{ background: #262a33; border-left: 3px solid {ACCENT}; color: white; }}
QListWidget#nav::item:hover {{ background: #22252c; }}
QGroupBox {{ border: 1px solid #353a45; border-radius: 6px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9fb6ff; font-weight: bold; }}
QPushButton {{ padding: 5px 12px; border-radius: 4px; background: #2d323c; border: 1px solid #424857; }}
QPushButton:hover {{ background: #363c48; }}
QPushButton:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton:disabled {{ color: #6b7180; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: bold; padding: 7px 20px; }}
QPushButton#primary:hover {{ background: #5a9dff; }}
QPushButton#danger {{ background: #b33a3a; border-color: #b33a3a; color: white; font-weight: bold; padding: 7px 20px; }}
QToolButton {{ padding: 4px 8px; border-radius: 4px; }}
QToolButton:checked {{ background: {ACCENT}; color: white; }}
QProgressBar {{ border: 1px solid #424857; border-radius: 4px; text-align: center; background: #1f2228; height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QLabel#h1 {{ font-size: 16pt; font-weight: bold; color: white; }}
QLabel#hint {{ color: #9aa1ad; }}
QCheckBox#cls {{ padding: 2px; }}
QTableWidget, QListWidget, QListView, QPlainTextEdit {{ background: #1c1f25; }}
QHeaderView::section {{ background: #262a33; padding: 4px; border: none; border-right: 1px solid #353a45; }}
"""


def apply_dark(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    base, alt, text = QColor(32, 35, 41), QColor(40, 44, 52), QColor(222, 225, 230)
    p.setColor(QPalette.ColorRole.Window, base)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, QColor(26, 28, 33))
    p.setColor(QPalette.ColorRole.AlternateBase, alt)
    p.setColor(QPalette.ColorRole.ToolTipBase, alt)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, alt)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor("red"))
    p.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 125, 135))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        p.setColor(QPalette.ColorGroup.Disabled, role, QColor(110, 115, 125))
    app.setPalette(p)
    app.setStyleSheet(STYLE)
