Thông tin sinh viên:
	Họ tên: Trần Quang Trung
	MSSV: 20154002

Mô tả cách làm:
	- Xây dựng hàm ridge với tham số đầu vào là dữ liệu.
	- Hàm ridge trả về giá trị sai số mse, trọng số w và lamda tìm được tìm được.
	- giá trị y_predict là giá trị dữ đoán của những điểm dữ liệu cần dự đoán với lamda và trọng số w tìm được.
	- Cách tìm lamda: 
		+ Tạo dải 100 000 giá trị trong khoảng (0, 100) cho lambda.
		+ Thử từng giá trị qua hàm rigde() và lưu lại kết quả lamda cho sai số đối với tập test là nhỏ nhất.
		+ Dựa vào kết quả đã thử nghiệm, tìm kết quả có giá trị mse nhỏ nhất.
		+ Từ đó rút ra giá trị lambda và kết quả dự đoán của x_test.
	=> kết quả lambda tìm được: 15.19516043200432
