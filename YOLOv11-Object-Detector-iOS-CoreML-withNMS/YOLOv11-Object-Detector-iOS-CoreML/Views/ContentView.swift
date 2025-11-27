
import SwiftUI
import PhotosUI
import UIKit
import AVFoundation

enum DetectionMode {
    case camera
    case photo
}

// MARK: - Text-to-Speech

final class SpeechManager: ObservableObject {
    private let synthesizer = AVSpeechSynthesizer()
    
    func speak(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        
        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = AVSpeechSynthesisVoice(language: "vi-VN")
        utterance.rate = 0.5
        utterance.pitchMultiplier = 1.0
        
        synthesizer.stopSpeaking(at: .immediate)
        synthesizer.speak(utterance)
    }
    
    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
    }
}

struct ContentView: View {
    let cameraService = CameraService()
    
    @Bindable private var detector: Detector
    @Bindable private var mainDishClassifier: MainDishClassifier
    
    @State private var mode: DetectionMode = .camera
    @State private var selectedImage: UIImage?
    @State private var photoItem: PhotosPickerItem?

    
    @StateObject private var speechManager = SpeechManager()
    
    init() {
        self.detector = Detector(modelName: Constants.modelName)
        self.mainDishClassifier = MainDishClassifier(modelName: Constants.mainDishModelName)
    }
    
    // MARK: - Helper: đếm số lượng từng nhãn
    
    private var objectCounts: [(label: String, count: Int)] {
        let labels = detector.detectedObjects.map { $0.label }
        let dict = Dictionary(grouping: labels, by: { $0 }).mapValues { $0.count }
        return dict
            .map { (label: $0.key, count: $0.value) }
            .sorted { $0.label < $1.label }
    }
    
    // Màu ổn định cho mỗi label (hash giống BoundingBoxView)
    private func color(for label: String) -> Color {
        var hasher = Hasher()
        hasher.combine(label)
        let hash = hasher.finalize()
        let r = Double((hash & 0xFF0000) >> 16) / 255.0
        let g = Double((hash & 0x00FF00) >> 8) / 255.0
        let b = Double(hash & 0x0000FF) / 255.0
        return Color(red: r, green: g, blue: b)
    }
    
    // Chuỗi đọc ra loa
    private func buildSpeechSentence() -> String {
        var parts: [String] = []
        
        if let rawDish = mainDishClassifier.mainDishLabel {
            let dish = LabelMapper.displayName(for: rawDish)
            let percent = Int(mainDishClassifier.mainDishConfidence * 100)
            parts.append("Món chính: \(dish), độ tin cậy \(percent) phần trăm.")
        }
        
        if !objectCounts.isEmpty {
            let ingredientText = objectCounts.map { item -> String in
                let prettyName = LabelMapper.displayName(for: item.label)
                if item.count > 1 {
                    return "\(item.count) \(prettyName)"
                } else {
                    return prettyName
                }
            }.joined(separator: ", ")
            
            parts.append("Thành phần gồm: \(ingredientText).")
        }
        
        if parts.isEmpty {
            return "Chưa nhận diện được món ăn hoặc thành phần rõ ràng."
        }
        
        return parts.joined(separator: " ")
    }
    
    // MARK: - Body
    
    var body: some View {
        ZStack {
            // Nền: camera hoặc ảnh
            Group {
                switch mode {
                case .camera:
                    CameraView(cameraService: cameraService) { buffer in
                        detector.detectObjects(pixelBuffer: buffer)
                        mainDishClassifier.classify(pixelBuffer: buffer)
                    }
                case .photo:
                    GeometryReader { geometry in
                        if let image = selectedImage {
                            ZStack {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(width: geometry.size.width,
                                           height: geometry.size.height)
                                    .background(Color.black)
                                
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
            
            // Overlay box khi camera
            if mode == .camera {
                GeometryReader { geometry in
                    ForEach(detector.detectedObjects) { object in
                        BoundingBoxView(object: object,
                                        parentSize: geometry.size)
                    }
                }
            }
            
            // Overlay UI
            VStack {
                headerView()
                    .padding(.top, 40)
                    .padding(.horizontal, 16)
                
                Spacer()
                
                bottomPanel()
            }
        }
        .background(Color.black)
        .ignoresSafeArea()
        .onDisappear {
            speechManager.stop()
        }
        .onChange(of: photoItem) { newItem in
            guard let newItem else { return }
            
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data),
                   let cgImage = uiImage.cgImage {
                    
                    await MainActor.run {
                        self.selectedImage = uiImage
                        self.mode = .photo
                        self.detector.detectedObjects = []
                    }
                    
                    detector.detectObjects(on: cgImage)
                    mainDishClassifier.classify(cgImage: cgImage)
                }
            }
        }
    }
    
    // MARK: - Header
    
    @ViewBuilder
    private func headerView() -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Bùi Quang Thái: Nhận diện nguyên liệu")
                    .font(.headline)
                    .bold()
                    .foregroundColor(.white)
                
                Text("Đưa camera vào bàn ăn để nhận diện thành phần của món ăn")
                    .font(.caption)
                    .foregroundColor(.white.opacity(0.9))
                
                if let rawDish = mainDishClassifier.mainDishLabel {
                    let dish = LabelMapper.displayName(for: rawDish)
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
    }
    
    // MARK: - Bottom Panel (button cố định dưới cùng)
    
    @ViewBuilder
    private func bottomPanel() -> some View {
        VStack(spacing: 10) {
            // Hai nút chọn mode
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
            
            // Khối thống kê + thời gian (mọc lên từ dưới)
            if !objectCounts.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Thống kê thành phần trong món ăn")
                        .font(.subheadline)
                        .bold()
                        .foregroundColor(.white)
                    
                    ForEach(objectCounts, id: \.label) { item in
                        let prettyName = LabelMapper.displayName(for: item.label)
                        HStack(spacing: 6) {
                            Circle()
                                .fill(color(for: item.label))
                                .frame(width: 8, height: 8)
                            
                            Text("\(prettyName): \(item.count)")
                                .font(.caption2)
                                .foregroundColor(.white)
                        }
                    }
                }
                .padding(8)
                .background(Color.black.opacity(0.7))
                .cornerRadius(10)
            }
            
            Text(String(format: "Thời gian phản hồi: %.1f ms", detector.interfaceTime))
                .font(.footnote)
                .foregroundColor(.white)
                .padding(6)
                .background(Color.black.opacity(0.6))
                .cornerRadius(8)
            
            // Nút ĐỌC – luôn là phần cuối cùng, dính sát đáy
            Button {
                let sentence = buildSpeechSentence()
                speechManager.speak(sentence)
            } label: {
                HStack {
                    Image(systemName: "speaker.wave.2.fill")
                    Text("Đọc món & thành phần")
                }
                .font(.subheadline)
                .padding(.vertical, 10)
                .padding(.horizontal, 18)
                .frame(maxWidth: .infinity)
                .background((mainDishClassifier.mainDishLabel == nil && objectCounts.isEmpty) ? Color.gray.opacity(0.5) : Color.orange)
                .foregroundColor(.white)
                .cornerRadius(16)
            }
            .disabled(mainDishClassifier.mainDishLabel == nil && objectCounts.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 16)
        .background(
            LinearGradient(
                gradient: Gradient(colors: [Color.black.opacity(0.7), Color.black.opacity(0.0)]),
                startPoint: .bottom,
                endPoint: .top
            )
            .ignoresSafeArea(edges: .bottom)
        )
    }
}

#Preview {
    ContentView()
}
