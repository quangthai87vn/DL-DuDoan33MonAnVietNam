
import Foundation

/// Map tên lớp YOLO (không dấu) → tên hiển thị tiếng Việt có dấu
enum LabelMapper {
    /// CHỈ CẦN SỬA LIST NÀY NẾU SAU NÀY ĐỔI NHÃN
    static let mapping: [String: String] = [
        // ===== MÓN / THÀNH PHẦN =====
        "banh_da": "Bánh đa",
        "banh_gao": "Bánh gạo",
        "banh_gao_cay_tokbokki": "Bánh gạo cay Tteokbokki",
        "banh_hoi": "Bánh hỏi",
        "banh_hoi_heo_quay": "Bánh hỏi heo quay",
        "banh_hu_tieu": "Bánh hủ tiếu",
        "banh_mi_sandwish": "Bánh mì sandwich",
        "banh_pho": "Bánh phở",
        "bap": "Bắp",
        "bi": "Bì",
        "bun": "Bún",
        "bun_bo": "Bún bò",
        "bun_ca": "Bún cá",
        "bun_dau": "Bún đậu",
        "bun_hai_san": "Bún hải sản",
        "bun_moc": "Bún mọc",
        "ca": "Cá",
        "ca_hoi": "Cá hồi",
        "ca_hoi_sot_pho_mai_kem": "Cá hồi sốt phô mai kem",
        "cac_loai_rau_cu": "Các loại rau củ",
        "cha": "Chả",
        "cha_ca": "Chả cá",
        "cha_com": "Chả cốm",
        "cha_gio": "Chả giò",
        "com": "Cơm",
        "com_ca_ri": "Cơm cà ri",
        "com_chien_cai_xoan": "Cơm chiên cải xoăn",
        "com_ga": "Cơm gà",
        "com_suon": "Cơm sườn",
        "com_suon_bi_cha": "Cơm sườn bì chả",
        "com_suon_bi_cha_trung": "Cơm sườn bì chả trứng",
        "com_suon_bi_trung": "Cơm sườn bì trứng",
        "com_suon_cha": "Cơm sườn chả",
        "com_suon_trung": "Cơm sườn trứng",
        "dau_hu": "Đậu hũ",
        "doi_sun": "Dồi sụn",
        "dua_chua": "Dưa chua",
        "ga_ran_va_khoai_tay_chien": "Gà rán và khoai tây chiên",
        "gan": "Gan",
        "hu_tieu": "Hủ tiếu",
        "hu_tieu_mi": "Hủ tiếu mì",
        "khoai_tay_chien": "Khoai tây chiên",
        "mi": "Mì",
        "mi_ga_hung_que_kem": "Mì gà húng quế kem",
        "mi_jajangmyeon": "Mì Jajangmyeon",
        "mi_misoramen": "Mì miso ramen",
        "mi_nuoc_sup_bun_bo": "Mì nước súp bún bò",
        "mi_quang": "Mì Quảng",
        "mi_tron": "Mì trộn",
        "mi_tuong_den_jajangmyeon": "Mì tương đen Jajangmyeon",
        "muc": "Mực",
        "naruto": "Naruto",
        "pho_bo": "Phở bò",
        "pho_ga": "Phở gà",
        "pho_mai": "Phô mai",
        "rau_ram": "Rau răm",
        "soi_mi_khoai_lang": "Sợi mì khoai lang",
        "soi_mi_miso": "Sợi mì miso",
        "sot_tuong_den": "Sốt tương đen",
        "suon": "Sườn",
        "thit_bam": "Thịt bằm",
        "thit_bo": "Thịt bò",
        "thit_ga": "Thịt gà",
        "thit_heo": "Thịt heo",
        "thit_heo_chien_xu_tonkatsu": "Thịt heo chiên xù Tonkatsu",
        "thit_heo_quay": "Thịt heo quay",
        "thit_lon_bistek": "Thịt lợn Bistek",
        "tom": "Tôm",
        "tom_tit": "Tôm tít",
        "trung": "Trứng",
        "vien_moc": "Viên mọc",
        "xoi": "Xôi",
        "xoi_ga_chien": "Xôi gà chiên"
    ]
    
    /// Chuẩn hoá label YOLO → key dạng `banh_da`
    private static func normalizedKey(from raw: String) -> String {
        raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
    }
    
    /// Trả về tên hiển thị có dấu; nếu không có trong mapping thì fallback
    static func displayName(for raw: String) -> String {
        let key = normalizedKey(from: raw)
        
        if let mapped = mapping[key] {
            return mapped
        }
        
        // fallback: đổi _ thành space và viết hoa chữ cái đầu cho đỡ xấu
        let spaced = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: "_", with: " ")
        return spaced.capitalized
    }
}
