"""Image canvas with zone drawing (ignore / include polygons) and a timestamp region."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsPolygonItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView)

from ..core.classes import color_for
from ..core.config import Zone
from .common import bgr_to_pixmap

ZONE_STYLE = {
    "ignore": (QColor(255, 70, 70), QColor(255, 70, 70, 70), "IGNORE"),
    "include": (QColor(60, 220, 120), QColor(60, 220, 120, 55), "ONLY HERE"),
}


class ZoneItem(QGraphicsPolygonItem):
    def __init__(self, kind, poly: QPolygonF):
        super().__init__(poly)
        self.kind = kind
        line, fill, label = ZONE_STYLE[kind]
        pen = QPen(line, 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(fill))
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setZValue(5)
        t = QGraphicsSimpleTextItem(label, self)
        t.setBrush(QBrush(line))
        f = t.font()
        f.setBold(True)
        f.setPointSize(10)
        t.setFont(f)
        t.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        b = poly.boundingRect()
        t.setPos(b.left() + 4, b.top() + 2)

    def scene_points(self):
        return [self.mapToScene(p) for p in self.polygon()]


class ZoneCanvas(QGraphicsView):
    changed = Signal()
    status = Signal(str)

    TOOLS = ("pan", "ignore-rect", "ignore-poly", "include-rect", "include-poly", "timestamp")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(18, 19, 23))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.pix = QGraphicsPixmapItem()
        self.pix.setZValue(0)
        self.scene().addItem(self.pix)
        self.w = self.h = 1
        self.tool = "pan"
        self._drag_start = None
        self._temp = None
        self._poly_pts: list[QPointF] = []
        self.ts_item = None
        self.det_items = []
        self._user_zoomed = False
        self.setMinimumSize(400, 260)
        self.setMouseTracking(True)

    # ----- image -----
    def set_image(self, img):
        if img is None:
            return
        h, w = img.shape[:2]
        resized = (w, h) != (self.w, self.h)
        self.w, self.h = w, h
        self.pix.setPixmap(bgr_to_pixmap(img))
        self.scene().setSceneRect(QRectF(0, 0, w, h))
        if resized or not self._user_zoomed:
            self.fit()

    def fit(self):
        self._user_zoomed = False
        self.fitInView(QRectF(0, 0, self.w, self.h), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._user_zoomed:
            self.fit()

    def wheelEvent(self, e):
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(f, f)
        self._user_zoomed = True

    # ----- tools -----
    def set_tool(self, tool):
        self._cancel_temp()
        self.tool = tool
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tool == "pan" else QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if tool == "pan" else Qt.CursorShape.CrossCursor)
        hints = {
            "pan": "Drag to pan, wheel to zoom. Click a zone to select it, Delete removes it, drag moves it.",
            "ignore-rect": "Drag a rectangle: detections inside are IGNORED.",
            "include-rect": "Drag a rectangle: ONLY detections inside are kept.",
            "ignore-poly": "Click points to outline an IGNORE area. Double-click or Enter to finish, Esc cancels.",
            "include-poly": "Click points to outline an ONLY-HERE area. Double-click or Enter to finish, Esc cancels.",
            "timestamp": "Drag a rectangle around the burned-in date/time text (optional; auto-detect otherwise).",
        }
        self.status.emit(hints.get(tool, ""))

    def _scene_pt(self, e):
        p = self.mapToScene(e.position().toPoint())
        return QPointF(min(max(p.x(), 0), self.w), min(max(p.y(), 0), self.h))

    def mousePressEvent(self, e):
        if self.tool == "pan" or e.button() != Qt.MouseButton.LeftButton:
            if e.button() == Qt.MouseButton.RightButton and self._poly_pts:
                self._poly_pts.pop()
                self._update_poly_temp(None)
                return
            return super().mousePressEvent(e)
        p = self._scene_pt(e)
        if self.tool.endswith("-poly"):
            if len(self._poly_pts) >= 3 and _dist(self.mapFromScene(p), self.mapFromScene(self._poly_pts[0])) < 10:
                self._finish_poly()
            else:
                self._poly_pts.append(p)
                self._update_poly_temp(p)
        else:
            self._drag_start = p
            self._temp = QGraphicsRectItem()
            pen = QPen(QColor("yellow" if self.tool == "timestamp" else "white"), 1, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._temp.setPen(pen)
            self._temp.setZValue(10)
            self.scene().addItem(self._temp)

    def mouseMoveEvent(self, e):
        if self._drag_start is not None and self._temp is not None:
            self._temp.setRect(QRectF(self._drag_start, self._scene_pt(e)).normalized())
        elif self.tool.endswith("-poly") and self._poly_pts:
            self._update_poly_temp(self._scene_pt(e))
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag_start is not None and self._temp is not None:
            r = self._temp.rect()
            self.scene().removeItem(self._temp)
            self._temp, self._drag_start = None, None
            if r.width() > 4 and r.height() > 4:
                if self.tool == "timestamp":
                    self.set_ts_region([r.left() / self.w, r.top() / self.h, r.right() / self.w, r.bottom() / self.h])
                else:
                    kind = self.tool.split("-")[0]
                    self._add_zone(kind, QPolygonF([r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft()]))
                self.changed.emit()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.tool.endswith("-poly") and len(self._poly_pts) >= 3:
            self._finish_poly()
            return
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and len(self._poly_pts) >= 3:
            self._finish_poly()
        elif e.key() == Qt.Key.Key_Escape:
            self._cancel_temp()
        elif e.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        else:
            super().keyPressEvent(e)

    def _update_poly_temp(self, cursor):
        if self._temp is None:
            self._temp = QGraphicsPathItem()
            pen = QPen(QColor("white"), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._temp.setPen(pen)
            self._temp.setZValue(10)
            self.scene().addItem(self._temp)
        path = QPainterPath()
        if self._poly_pts:
            path.moveTo(self._poly_pts[0])
            for p in self._poly_pts[1:]:
                path.lineTo(p)
            if cursor is not None:
                path.lineTo(cursor)
        self._temp.setPath(path)

    def _finish_poly(self):
        kind = self.tool.split("-")[0]
        self._add_zone(kind, QPolygonF(self._poly_pts))
        self._cancel_temp()
        self.changed.emit()

    def _cancel_temp(self):
        if self._temp is not None:
            self.scene().removeItem(self._temp)
        self._temp, self._drag_start, self._poly_pts = None, None, []

    def _add_zone(self, kind, poly):
        it = ZoneItem(kind, poly)
        self.scene().addItem(it)
        return it

    def delete_selected(self):
        sel = [i for i in self.scene().selectedItems() if isinstance(i, ZoneItem) or i is self.ts_item]
        for it in sel:
            if it is self.ts_item:
                self.ts_item = None
            self.scene().removeItem(it)
        if sel:
            self.changed.emit()

    # ----- zones API -----
    def zone_items(self):
        return [i for i in self.scene().items() if isinstance(i, ZoneItem)]

    def get_zones(self) -> list[dict]:
        out = []
        for it in reversed(self.zone_items()):
            pts = [[round(p.x() / self.w, 5), round(p.y() / self.h, 5)] for p in it.scene_points()]
            out.append({"kind": it.kind, "points": pts})
        return out

    def set_zones(self, zones):
        for it in self.zone_items():
            self.scene().removeItem(it)
        for z in zones:
            z = z if isinstance(z, Zone) else Zone.from_dict(z)
            if len(z.points) >= 3:
                self._add_zone(z.kind, QPolygonF([QPointF(x * self.w, y * self.h) for x, y in z.points]))

    def clear_zones(self):
        self.set_zones([])
        self.set_ts_region(None)
        self.changed.emit()

    def set_ts_region(self, r):
        if self.ts_item is not None:
            self.scene().removeItem(self.ts_item)
            self.ts_item = None
        if r:
            x1, y1, x2, y2 = r
            it = QGraphicsRectItem(QRectF(x1 * self.w, y1 * self.h, (x2 - x1) * self.w, (y2 - y1) * self.h))
            pen = QPen(QColor(255, 220, 0), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            it.setPen(pen)
            it.setBrush(QBrush(QColor(255, 220, 0, 40)))
            it.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            it.setZValue(6)
            lbl = QGraphicsSimpleTextItem("TIMESTAMP", it)
            lbl.setBrush(QBrush(QColor(255, 220, 0)))
            lbl.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            lbl.setPos(x1 * self.w + 2, y1 * self.h + 2)
            self.scene().addItem(it)
            self.ts_item = it

    def get_ts_region(self):
        if self.ts_item is None:
            return None
        r = self.ts_item.rect()
        return [round(r.left() / self.w, 5), round(r.top() / self.h, 5),
                round(r.right() / self.w, 5), round(r.bottom() / self.h, 5)]

    # ----- detections overlay -----
    def set_detections(self, kept=(), dropped=(), names=None):
        for it in self.det_items:
            self.scene().removeItem(it)
        self.det_items = []
        names = names or {}
        for rows, active in ((dropped, False), (kept, True)):
            for r in rows:
                x1, y1, x2, y2, conf, c = r[:6]
                col = QColor(*reversed(color_for(int(c)))) if active else QColor(150, 150, 150)
                pen = QPen(col, 2 if active else 1, Qt.PenStyle.SolidLine if active else Qt.PenStyle.DashLine)
                pen.setCosmetic(True)
                it = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
                it.setPen(pen)
                it.setZValue(8)
                self.scene().addItem(it)
                txt = QGraphicsSimpleTextItem(f"{names.get(int(c), int(c))} {conf:.2f}" + ("" if active else " (filtered)"), it)
                txt.setBrush(QBrush(col))
                txt.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                txt.setPos(x1, max(0, y1 - 2))
                self.det_items.append(it)


def _dist(a, b):
    return ((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2) ** 0.5
