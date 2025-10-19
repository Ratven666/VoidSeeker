import ezdxf

from app.scan.ScanPoint import ScanPoint
from app.scan.parsers.ScanParserABC import ScanParserABC


class ScanParserFromDxf(ScanParserABC):

    def parse(self, scan):
        doc = ezdxf.readfile(self.file_path)
        msp = doc.modelspace()
        for entity in msp:
            # Если объект является точкой
            if entity.dxftype() == 'POINT':
                location = entity.dxf.location  # Координаты точки (x, y, z)
                point = ScanPoint(*location)
                scan.add_point(point)
