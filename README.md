# Phân loại và phân đoạn 33 món ăn Việt Nam
## Summary
- Trong dự án này, tôi sử dụng Pytorch để thực hiện phân loại và phân đoạn 33 món ăn phổ biến ở Việt Nam. Ngoài ra, để việc phân đoạn ảnh dễ dàng hơn, tôi đã sử dụng thêm thư viện [Segmentation Pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Tập dữ liệu :egg: 
- Dữ liệu 30 món ăn được lấy từ tập dữ liệu [30VNFoods](https://www.kaggle.com/datasets/quandang/vietnamese-foods) và thêm 3 món ăn mới bao gồm : Bánh da lợn, bánh tiêu, bánh trung thu

## Mô hình
- Tôi sử dụng nhiều mô hình khác nhau, từ MLP đến CNN đơn giản, miniVGG. Các mô hình được đào tạo trước như VGG16, ResNet18.
- Đối với bài toán Phân đoạn, tôi sử dụng cấu trúc Unet với các bộ mã hóa là các mô hình được đào tạo trước để có được kết quả tốt nhất.
- Tôi sử dụng Wandb để theo dõi và so sánh các thí nghiệm: [Classification](https://wandb.ai/harly/classifi_FoodVN?workspace=user-harly), [Segmentation](https://wandb.ai/harly/SegVNFood?workspace=user-harly)

## Cách chạy  Docker
```python
docker stop vnfoods-app 
docker rm vnfoods-app 
sudo docker build -t vnfoods-app:latest .
docker run -d --name vnfoods-app --gpus all -p 6789:6789 \
  -v "/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images:/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images:ro" \
  -v "$PWD/Jupyter/runs:/app/Jupyter/runs:ro" \
  -v "$PWD/Jupyter/images:/app/Jupyter/images:ro" \
  -e DATA_DIR="/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images" \
  vnfoods-app
```


## Cách chạy Project :question:
```python
git clone https://github.com/quangthai87vn/DL-DuDoan33MonAnVietNam.git
cd DL-DuDoan33MonAnVietNam
pip install -r requirements.txt
# Huấn luyện mô hình CNN tự xây dựng
python classifi_main.py --model cnn --epochs 100 --batch_size 64
python classifi_main.py --model mtl_cnn --epochs 100 --batch_size 64

# Huấn luyện mô hình efficientnet_b0
python classifi_main.py --model efficientnet_b0 --epochs 100 --batch_size 64

# Huấn luyện mô hình mtl-efficientnet_b0: 87.76%
python classifi_main.py --model mtl_efficientnet_b0 --epochs 100 --batch_size 64

# Huấn luyện mô hình VGG16 / ResNet18 (nếu muốn)
python classifi_main.py --model vgg16 --epochs 100
python classifi_main.py --model resnet18 --epochs 100


#run segmentation
python seg_main.py
# khi triển khai trên Docker để chạy UI App dự đoán thì file docker tự kích hoạt, chạy Local thì run code sau

streamlit run .\app.py
```
# Huấn luyện và kiểm tra mô hình MobilenetV4
```bash
python classifi_main.py --model mobilenetv4 --epochs 100 --batch_size 64
```
Kiểm tra mô hình Mobinet
```bash
python mobilenet_test.py --image_path C:\Users\Admin\OneDrive\Desktop\3.jpg --model_path C:\Users\Admin\OneDrive\DOCKER\Apps\DL-DuDoan33MonAnVietNam\Models\MTL-MobileNet.pth --label_path label.txt
```





python train_finetune_efficientnet.py \
  --data_dir "/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods" \
  --epochs 21000 \
  --batch_size 32 \
  --img_size 256 \
  --freeze_until 0 \
  --mixed_precision \
  --outdir runs/effb0_256

 
python classifi_main.py --model mtl_cnn --epochs 100 --batch_size 64
python classifi_main.py --model efficientnet_b0 --epochs 100 --batch_size 64
python classifi_main.py --model vgg16 --epochs 100



**__Lưu ý__**: Khi bạn chạy seg_main.py, phải mất 8 đến 10 phút để chuẩn bị dữ liệu
## Kết quả phân loại
|     Methods                |     Accuracy    |     Loss        |     Val_Accuracy    |     Val_Loss    |     Test_accuracy    |
|----------------------------|-----------------|-----------------|---------------------|-----------------|----------------------|
|     Resnet18_pretrained    |     99.926      |     6.78E-05    |     96.907          |     0.1106      |     95.886           |
|     Resnet18               |     99.486      |     0.0003      |     80.154          |     0.7141      |     78.663           |
|     VGG16_pretrained       |     99.266      |     0.0005      |     94.587          |     0.4035      |     95.758           |
|     VGG16                  |     95.229      |     0.0030      |     78.350          |     0.6939      |     77.763           |
|     miniVGG                |     99.926      |     0.0001      |     82.989          |     0.6325      |     87.917           |
|     SimpleCNN              |     99.559      |     0.0008      |     86.597          |     0.3855      |     86.632           |
|     MLP_4hidden512node     |     53.651      |     0.0678      |     45.103          |     2.8904      |     47.043           |
|     MLP_3hidden1024node    |     44.403      |     0.1080      |     34.278          |     4.8297      |     38.946           |
|     MLP_3hidden512node     |     55.486      |     0.0707      |     40.721          |     5.5563      |     44.987           |
|     MLP_4hidden            |     47.706      |     0.0583      |     37.886          |     2.3706      |     38.303           |
|     MLP_3hidden            |     49.761      |     0.0512      |     36.082          |     3.0187      |     41.902           |
|     MLP_2hidden            |     48.844      |     0.0438      |     40.979          |     1.6916      |     41.516           |
## Kết quả phân đoạn
|     Methods          |     iou/valid    |     iou     banhmi    |     iou     banhtrang    |     iou     comtam    |     iou     pho    |     iou_clutter    |
|----------------------|------------------|-----------------------|--------------------------|-----------------------|--------------------|--------------------|
|     Unet_ResNet34    |     0.8625       |     0.8273            |     0.8529               |     0.7083            |     0.7099         |     0.9084         |
|     Unet-ResNet18    |     0.8828       |     0.8655            |     0.8897               |     0.7893            |     0.7571         |     0.9214         |
|     Unet-VGG16       |     0.8716       |     0.8627            |     0.8713               |     0.7395            |     0.7463         |     0.9146         |
## Plot Val Accuracy
- Classification:
![Example](https://github.com/quangthai87vn/DL-DuDoan33MonAnVietNam/blob/main/images/W%26B%20valac.png "This is a sample image.")
- Segmentation:
![image](https://github.com/Harly-1506/4VNfoods-Deep-learning/assets/86733695/6d772489-a7a4-47b6-b6e9-5fe7da503fd3)

## Demo:
- Demo trong VSCode: streamlit run app.py
- Chương trình demo bạn có thể theo dõi trong kho lưu trữ này: [Demo](http://mtltechnology.ddns.net:1111/)
___

*Hãy cho tôi một ngôi sao :star: nếu bạn thấy nó hữu ích, cảm ơn*







Kết hợp những yêu cầu nãy giờ hãy train mô hình "mtl_effcientnet_b0" với các thông số tốt nhất
Yêu cầu chính
- Code giải thích chi tiết , dể hiểu, các thông số  biến toàn cục và khai báo trên cùng, các hàm phụ nếu được sẽ truyền parameter để sử dụng lại
- Sử dụng CUDA với VGA RTX 5000 16GB,epochs 100, batch_size 64, Dẹp warning PIL & chuẩn hoá RGB
- Cấu trúc tập hình ảnh như sau 
DATA_DIR = Path("/media/mtl/DATA 6TB/AI DATASET/vietnamese-foods/Images")
TRAIN_DIR = DATA_DIR/"Train"
VAL_DIR   = DATA_DIR/"Validate"
TEST_DIR  = DATA_DIR/"Test"
- Ngừng train sau 5 lần không thấy cải thiện độ chính xác
- Cứ mỗi epoch train xong thì lưu ra 2 file có tên "mtl_effcientnet_b0_best" , "mtl_effcientnet_b0_last", lưu file CSV độ chính xác mỗi Epoch, bao gồm cả thời gian train 1 Epoch
- Sau khi train xong hoàn toàn thì thì vẽ biểu đồ độ chính xác, và lưu thành file ảnh
- Sau đó đánh giá mô hình bằng các biểu đồ    . 
- Đặc bịêt confusion matrix thì cho tuỳ chọn độ sai lệch dưới bao nhiêu % thì vẻ số lượng sai và % sai.  cho DPI của ảnh hiển thị confusion matrix lên 600, số phần trăm làm tròn thành số nguyên, mổi ô nếu có hiển thị thì hiển thị 2 dòng (dòng trên là số dự đoán sai, dòng dưới là số $ sai theo mỗi lớp). Vẽ hình ảnh sao cho trực quan nhất và dể nhìn
- Lựa chọn ngẫu nhiên với tuỳ chọn n ãnh (truyền vào) các file hình ảnh trong tập test và dự đoán xem độ chính xác bao nhiêu
- Lưu ý chung . với mỗi model khi train sẽ nằm trong thư mục "runs" và tạo ra thêm 1 folder riêng (ví dụ "mtl_efficientnet_{datetime}}") . folder riêng này sẽ tạo ra thêm thư mục "images" để lưu các hình ảnh đánh giá mô hình tại đây sau khi xuất ra

Tôi sẽ yêuc cầu thêm. Tạo lại toàn bộ cell mới nhé


Cần App Streamlit làm các nhiệm vụ sao, Các tab thiết kế trực quan để dể dàng bổ sung thêm Tab mới sau này
- Tạo Folder riêng là Webapps và lưu các file UI vào trong này
- File giao diện bao gồm 1 file chính app.py load các module chức năng. Trên giao diện chính gi8ới thiệu bài toán, hướng xữ lý
- Các Tab (module)) là file file riêng lẻ
- Thiết kế dạng Tab làm theo các module chức nămg
1. Module khai phá dữ liệu: file riêng app_datamining.py
+ Biểu đồ phân bố số lượng ảnh Train / Val / Test
+ Phân phối dữ liệu tập huấn luyện, kiểm tra và validate
+ Thống kê số lượng ảnh theo từng món ăn (Train / Validate / Test)
1. Module dự đoán ảnh (bạn đã làm) file riêng app_predict.py
3. Module Đánh giá model: file riêng app_validatemodel.py




Đây là 4 file đang chạy web streamlit ok, làm thế nào check lại 1 lần nữa yêu cầu
- Tiêu đề ghi là: "Nhận diện 33 món ăn - Bùi Quang Thái - 24752551"
- Bố trí dạng Tab phía bên tay trái kèm mô tả, bên phải là hiển thị nội dung Tab đó khi bấm vào
- Các hàm hay biến dùng dung chỉ load 1 lần, tránh trùng lắp gây nặng
- Không load ngay các biểu đồ, mà chỉ hiện nút button ví dụ "Loa biểu đồ Confumation Matrix" khi đó user bấm vào mới xữ lý và load cho nhẹ (làm y chang cho 3 Module trên)
- Kể từ lần sau khi tôi nói thêm các biểu đồ hay đánh giá thì vẩn theo cấu trúc là hiển thị nút nhấn, nhấn vào thì mới xữ lý
Bố trí trực quan dể  nhìn, chạy trên Mobile đẹp, màu sắc cân đối
- Xữ lý xong gửi fullcode 4 file trên để chép về



theo dòng, bấm vào thì vẽ ảnh dưới button đó, buton kế tiếp cụng vậy, làm y chang cấu trúc này cho toàn bộ file


Giúp tôi triển khai ứng dụng này bằng Docker có tích hợp Stremlit 


giãi thích này là trong code (ok giữ nguyên). Tôi muôn phần giải thích hiện ra ngoài UI luôn cho người dùng hiểu ý nghĩa của biểu đồ. dúng st.title và st.expander , st.markdown để làm việc này nhé, thiết kế UI cho đẹp


Trong 1 quy trình xây dựng model học máy thì gồm bao nhiêu bước
Có vẽ như tôi đang làm 3 bước rồi. Hãy gợi ý tôi xem nếu App Streamlit tôi cần bổ sung các bước còn lại ko và bổ sung như thế nào
Đặc biêt bước xây dựng mô hình nên làm thế nào  , tôi gửi 2 file lúc train để biết các dữ liệu dc làm như thế nào để bạn có cái nhìn tỗng quan xữ lý 




Thiết kế từ đầu cho tôi file yolo.ipynb đặt trong thư mục "/Jupyter/yolo.ipynb"
File model train Classifire nhận diện ảnh 33 món đang nằm trong "/Jupyter/runs/mtl_efficientnet_b0_01/checkpoints/mtl_effcientnet_b0_best.pt"
File model cấu trúc effcientnet_b0 đang được lư tại "/model/mtl_efficientnet_b0.py" 
Thư mục model cùng cấp với thư mục Jupyter
- Yêu cầu: viết từng cell trên Jupyter trong VSCode từ import thư viện + load mô hình + dự đoán + truyền ảnh thật và phân vùng các vật thể trong ảnh là món ăn, và món nào biết tên thì hiện tên lên
- Load đúng trọng số hay ClassName trong model sau khi train và check lại cho đúng



Xem cách load model trong file stremlit để  load đúng trọng số  và Classname 


Nếu muốn segmentation cực chuẩn + đúng từng món → phải chơi hẳn một bài train model segmentation riêng cho dataset món ăn. Hình ảnh là rất nhiều món ăn tôi chỉ cần segmentation vào món ăn thôi cũng dc. Cho code segmentation train từ đầu


có thể nhận diện xong rồi lưu ảnh xuống giúp ko



thử nghĩ xem với tập dữ liệu 33 món ăn thì ta có thể  làm gì với Deep Learning cáo



Tôi cần làm báo cáo giải thích chi tiêt 1modell này (file tôi gửi kèm mtl_efficientnet_b0.py))
- Giải thích model này lấy từ đâu
- Số lớp , retrain thế nào, weight của imanet là sao
- Giải thích chi tiết cấu trúc model này

trĩển khai thuật toán nhận diện món ăn lại xem