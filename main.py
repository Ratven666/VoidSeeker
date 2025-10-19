from app.scan.Scan import Scan

scan = Scan(scan_name="TestScan")
print(scan)
scan.import_points_from_file(file_path="src/Камеры/data_3.dxf")
print(scan)


scan.plot()
