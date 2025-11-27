import SwiftUI
import UIKit
import AVFoundation

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
    
    @StateObject private var speechManager = SpeechManager()
    
    // Auto-speak state
    @State private var detectionBeganAt: Date? = nil
    @State private var lastAutoSentence: String = ""
    @State private var lastAutoSpeakTime: Date = .distantPast
    @State private var hasSpokenAuto: Bool = false   // đã auto đọc cho cảnh hiện tại chưa
    
    init() {
        self.detector = Detector(modelName: Constants.modelName)
        self.mainDishClassifier = MainDishClassifier(modelName: Constants.mainDishModelName)
    }
    
    // MARK: - Helpers
    
    /// Đếm số lượng từng nhãn từ YOLO
    private var objectCounts: [(label: String, count: Int)] {
        let labels = detector.detectedObjects.map { $0.label }
        let dict = Dictionary(grouping: labels, by: { $0 }).mapValues { $0.count }
        return dict
            .map { (label: $0.key, count: $0.value) }
            .sorted { $0.label < $1.label }
    }
    
    /// Màu ổn định cho mỗi label (hash giống BoundingBoxView)
    private func color(for label: String) -> Color {
        var hasher = Hasher()
        hasher.combine(label)
        let hash = hasher.finalize()
        let r = Double((hash & 0xFF0000) >> 16) / 255.0
        let g = Double((hash & 0x00FF00) >> 8) / 255.0
        let b = Double(hash & 0x0000FF) / 255.0
        return Color(red: r, green: g, blue: b)
    }
    
    /// Câu sẽ đọc ra loa
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
    
    // MARK: - Auto-speak logic
    
    /// Xử lý mỗi frame từ camera
    private func processFrame(_ buffer: CVPixelBuffer) {
        detector.detectObjects(pixelBuffer: buffer)
        mainDishClassifier.classify(pixelBuffer: buffer)
        autoSpeakIfStable()
    }
    
    /// Tự đọc sau khi phát hiện ổn định ~3 giây, không spam
    private func autoSpeakIfStable() {
        let now = Date()
        
        // Không có món chính và không có thành phần → reset state
        if objectCounts.isEmpty && mainDishClassifier.mainDishLabel == nil {
            detectionBeganAt = nil
            hasSpokenAuto = false
            lastAutoSentence = ""
            return
        }
        
        // Nếu đã auto đọc cho cảnh hiện tại rồi thì thôi, không đọc nữa
        if hasSpokenAuto {
            return
        }
        
        // Bắt đầu đếm thời gian từ frame đầu tiên có detection
        if detectionBeganAt == nil {
            detectionBeganAt = now
            return
        }
        
        // Chưa đủ 3 giây thì đợi
        guard now.timeIntervalSince(detectionBeganAt!) >= 3 else { return }
        
        // Đủ 3 giây detection ổn định → build câu và đọc
        let sentence = buildSpeechSentence()
        guard !sentence.isEmpty else { return }
        
        speechManager.speak(sentence)
        hasSpokenAuto = true
        lastAutoSentence = sentence
        lastAutoSpeakTime = now
    }
    
    // MARK: - Body
    
    var body: some View {
        ZStack {
            // Nền: camera realtime
            CameraView(cameraService: cameraService) { buffer in
                processFrame(buffer)
            }
            
            // Vẽ bounding box
            GeometryReader { geometry in
                ForEach(detector.detectedObjects) { object in
                    BoundingBoxView(object: object,
                                    parentSize: geometry.size)
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
    
    // MARK: - Bottom Panel
    
    @ViewBuilder
    private func bottomPanel() -> some View {
        VStack(spacing: 10) {
            // Thống kê thành phần (mọc lên từ dưới)
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
            
            // Nút ĐỌC – người dùng chủ động bấm lại nếu muốn
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

