# Phân loại và phân đoạn 33 món ăn Việt Nam
## Summary
- Trong dự án này, tôi sử dụng Pytorch để thực hiện phân loại và phân đoạn 33 món ăn phổ biến ở Việt Nam. Ngoài ra, để việc phân đoạn ảnh dễ dàng hơn, tôi đã sử dụng thêm thư viện [Segmentation Pytorch](https://github.com/qubvel/segmentation_models.pytorch)

## Tập dữ liệu :egg: 
- Dữ liệu 30 món ăn được lấy từ tập dữ liệu [30VNFoods](https://www.kaggle.com/datasets/quandang/vietnamese-foods) và thêm 3 món ăn mới bao gồm : Bánh da lợn, bánh tiêu, bánh trung thu

   ![Example](https://github.com/quangthai87vn/DL-DuDoan33MonAnVietNam/blob/main/images/image.png "This is a sample image.")


## Mô hình
 - Tôi sử dụng nhiều mô hình khác nhau, từ MLP đến CNN đơn giản, miniVGG. Các mô hình được đào tạo trước như VGG16, ResNet18.

- Đối với bài toán Phân đoạn, tôi sử dụng cấu trúc Unet với các bộ mã hóa là các mô hình được đào tạo trước để có được kết quả tốt nhất.

- Tôi sử dụng Wandb để theo dõi và so sánh các thí nghiệm: [Classification](https://wandb.ai/harly/classifi_FoodVN?workspace=user-harly), [Segmentation](https://wandb.ai/harly/SegVNFood?workspace=user-harly)

## Cách chạy Project :question:
```python
git clone https://github.com/quangthai87vn/DL-DuDoan33MonAnVietNam.git
cd DL-DuDoan33MonAnVietNam
# train CNN cũ
python classifi_main.py --model cnn

# train MobileNet V1 mới
python classifi_main.py --model mobilenet

# train VGG16 / ResNet18 (nếu muốn)
python classifi_main.py --model vgg16
python classifi_main.py --model resnet18

#run segmentation
python seg_main.py

# khi triển khai trên Docker để chạy UI App dự đoán thì file docker tự kích hoạt, chạy Local thì run code sau
streamlit run .\app.py
```

Huấn luyện mô hình Mobinet

```bash
python mobilenet_train_enrnptys.py --train_dir C:/TRAIN/Deep Learning/vietnamese-foods/Images/Train --num_epochs 100 --batch_size 32 --model_path /models/MTL-MobileNet.pth
```

Kiểm tra mô hình Mobinet
```bash
python mobilenet_test.py --image_path C:\Users\Admin\OneDrive\Desktop\3.jpg --model_path C:\Users\Admin\OneDrive\DOCKER\Apps\DL-DuDoan33MonAnVietNam\Models\MTL-MobileNet.pth --label_path label.txt
```






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

- Chương trình demo bạn có thể theo dõi trong kho lưu trữ này: [Demo](http://mtltechnology.ddns.net:1111/)
___

*Hãy cho tôi một ngôi sao :star: nếu bạn thấy nó hữu ích, cảm ơn*
