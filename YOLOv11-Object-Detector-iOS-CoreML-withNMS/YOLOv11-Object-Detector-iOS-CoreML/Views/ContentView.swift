import SwiftUI
import PhotosUI
import UIKit

enum DetectionMode {
    case camera
    case photo
}

struct ContentView: View {
    let cameraService = CameraService()
    
    @Bindable private var detector: Detector               // YOLO – nguyên liệu
    @Bindable private var mainDishClassifier: MainDishClassifier // EfficientNet – món chính
    
    @State private var mode: DetectionMode = .camera
    @State private var selectedImage: UIImage?
    @State private var photoItem: PhotosPickerItem?
    
    init() {
        self.detector = Detector(modelName: Constants.modelName)
        self.mainDishClassifier = MainDishClassifier(modelName: Constants.mainDishModelName)
    }
    
    /// Đếm số lượng từng nhãn từ YOLO
    private var objectCounts: [(label: String, count: Int)] {
        let labels = detector.detectedObjects.map { $0.label }
        let dict = Dictionary(grouping: labels, by: { $0 }).mapValues { $0.count }
        return dict
            .map { (label: $0.key, count: $0.value) }
            .sorted { $0.label < $1.label }
    }
    
    var body: some View {
        ZStack {
            // Nền: camera hoặc ảnh
            Group {
                switch mode {
                case .camera:
                    // Realtime: camera + YOLO + món chính
                    CameraView(cameraService: cameraService) { buffer in
                        detector.detectObjects(pixelBuffer: buffer)
                        mainDishClassifier.classify(pixelBuffer: buffer)
                    }
                    
                case .photo:
                    GeometryReader { geometry in
                        if let image = selectedImage {
                            ZStack {
                                // ❗ Giữ nguyên tấm hình (không crop)
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: geometry.size.width,
                                           height: geometry.size.height)
                                    .background(Color.black)
                                
                                // Vẽ bounding box YOLO lên ảnh
                                ForEach(detector.detectedObjects) { object in
                                    BoundingBoxView(object: object,
                                                    parentSize: geometry.size)
                                }
                            }
                        } else {
                            Text("Chọn một hình để nhận diện")
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .background(Color.black)
                        }
                    }
                }
            }
            
            // Nếu đang ở camera thì overlay box riêng (tránh double vẽ)
            if mode == .camera {
                GeometryReader { geometry in
                    ForEach(detector.detectedObjects) { object in
                        BoundingBoxView(object: object,
                                        parentSize: geometry.size)
                    }
                }
            }
            
            // Overlay UI trên/dưới
            VStack {
                // HEADER
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Bùi Quang Thái: Nhận diện nguyên liệu")
                            .font(.headline)
                            .bold()
                            .foregroundColor(.white)
                        
                        Text("Đưa camera vào bàn ăn để nhận diện thành phần của món ăn")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.9))
                        
                        // Món chính từ EfficientNet – dùng cho cả camera & photo
                        if let dish = mainDishClassifier.mainDishLabel {
                            Text("Món chính: \(dish) (\(Int(mainDishClassifier.mainDishConfidence * 100))%)")
                                .font(.subheadline)
                                .bold()
                                .foregroundColor(.yellow)
                                .padding(.top, 4)
                        }
                    }
                    
                    Spacer()
                }
                .padding(10)
                .background(Color.black.opacity(0.6))
                .cornerRadius(12)
                .padding(.horizontal, 16)
                .padding(.top, 40)
                
                Spacer()
                
                // FOOTER: nút + thống kê
                VStack(alignment: .leading, spacing: 12) {
                    // 2 nút switch mode
                    HStack {
                        Button {
                            mode = .camera
                            selectedImage = nil
                        } label: {
                            HStack {
                                Image(systemName: "camera.fill")
                                Text("Nhận diện qua camera")
                            }
                            .font(.subheadline)
                            .padding(.vertical, 8)
                            .padding(.horizontal, 12)
                            .background(mode == .camera ? Color.white : Color.white.opacity(0.2))
                            .foregroundColor(mode == .camera ? .black : .white)
                            .cornerRadius(14)
                        }
                        
                        PhotosPicker(selection: $photoItem, matching: .images) {
                            HStack {
                                Image(systemName: "photo.on.rectangle")
                                Text("Nhận diện qua hình")
                            }
                            .font(.subheadline)
                            .padding(.vertical, 8)
                            .padding(.horizontal, 12)
                            .background(mode == .photo ? Color.white : Color.white.opacity(0.2))
                            .foregroundColor(mode == .photo ? .black : .white)
                            .cornerRadius(14)
                        }
                    }
                    
                    // Thống kê đối tượng
                    if !objectCounts.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Thống kê đối tượng")
                                .font(.subheadline)
                                .bold()
                                .foregroundColor(.white)
                            
                            ForEach(objectCounts, id: \.label) { item in
                                Text("\(item.label): \(item.count)")
                                    .font(.caption2)
                                    .foregroundColor(.white)
                            }
                        }
                        .padding(8)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(10)
                    }
                    
                    Text(String(format: "Thời gian phản hồi: %.1f ms", detector.interfaceTime))
                        .font(.footnote)
                        .foregroundColor(.white)
                        .padding(8)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(8)
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 16)
            }
        }
        .background(Color.black)
        .ignoresSafeArea()
        
        // Khi user chọn 1 tấm ảnh từ Photos
        .onChange(of: photoItem) { newItem in
            guard let newItem else { return }
            
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data),
                   let cgImage = uiImage.cgImage {
                    
                    // Cập nhật UI
                    await MainActor.run {
                        self.selectedImage = uiImage
                        self.mode = .photo
                        // clear kết quả cũ
                        self.detector.detectedObjects = []
                    }
                    
                    // YOLO: nguyên liệu
                    detector.detectObjects(on: cgImage)
                    // EfficientNet: món chính
                    mainDishClassifier.classify(cgImage: cgImage)
                }
            }
        }
    }
}

#Preview {
    ContentView()
}

