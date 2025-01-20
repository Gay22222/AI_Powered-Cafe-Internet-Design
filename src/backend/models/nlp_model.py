import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from underthesea import pos_tag
import pandas as pd
import os

# # Đặt thư mục tùy chỉnh cho dữ liệu NLTK (trong venv)
# nltk_data_dir = os.path.join(os.getcwd(), "venv", "nltk_data")
# if not os.path.exists(nltk_data_dir):
#     os.makedirs(nltk_data_dir)

# # Cấu hình NLTK để sử dụng thư mục này
# nltk.data.path.append(nltk_data_dir)
# nltk.download('punkt', download_dir=nltk_data_dir)
# nltk.download('punkt_tab', download_dir=nltk_data_dir)
# nltk.data.path = [nltk_data_dir]


class NLPModel:
    def __init__(self):
        self.furniture_keywords = ["phòng", "bàn", "ghế","lối", "quầy", "giữa"]
        self.default_parameters = {
            "room": {"type": "phòng", "size": "700x500", "unit": "cm"},
            "table": {"type": "bàn", "size": "120x60", "unit": "cm"},
            "chair": {"type": "ghế", "size": "60x60", "unit": "cm"},
            "distance_between_tables": {"type": "giữa", "size": "5", "unit": "cm"},
            "aisle_distance": {"type": "lối", "size": "98", "unit": "cm"},
            "reception_desk": {"type": "quầy", "size": "0x0", "unit": "cm", "present": "Không"}  # Có/Không
        }
        self.user_parameters = self.default_parameters.copy()

    def preprocess_text(self, text):
        """
        Chuẩn hóa văn bản: chuyển chữ thường, loại bỏ ký tự không cần thiết.
        """
        # text = self.resolve_coreference(text)
        text = text.lower().strip()
        text = ''.join([char for char in text if char.isalnum() or char.isspace() or char == 'x'])
        return text

    def tokenize_text(self, text):
        """
        Chia văn bản thành câu và token hóa.
        """
        sentences = sent_tokenize(text)
        tokens = [word_tokenize(sentence) for sentence in sentences]
        return sentences, tokens

    def extract_entities(self, text):
        """
        Nhận diện các thực thể từ văn bản đầu vào.
        """
        # Sử dụng Underthesea để gán nhãn từ loại
        words = pos_tag(text)
        print(f"Underthesea POS Tagging: {words}")  # Hiển thị kết quả gán nhãn từ loại
        
        detected_entities = []
        current_entity = None

        for word, tag in words:
            if word in self.furniture_keywords:
                if current_entity:
                    detected_entities.append(current_entity)
                    print(f"Entity Detected: {current_entity}")  # Theo dõi từng thực thể phát hiện
                current_entity = {"type": word, "value": ""}
            elif any(char.isdigit() for char in word) or "x" in word or "cm" in word or "m" in word:
                if current_entity:
                    current_entity["value"] += f"{word} "
        if current_entity:
            detected_entities.append(current_entity)
            print(f"Final Entity: {current_entity}")  # Hiển thị thực thể cuối cùng
        
        print(f"All Detected Entities: {detected_entities}")  # Theo dõi danh sách thực thể đầy đủ
        return pd.DataFrame(detected_entities)


    # def resolve_coreference(self, text):
    #     """
    #     Xử lý tham chiếu đồng đại, thay thế đại từ bằng thực thể gần nhất.
    #     """
    #     entities = {}
    #     sentences = sent_tokenize(text)
    #     resolved_text = []

    #     for sentence in sentences:
    #         words = word_tokenize(sentence)
    #         for i, word in enumerate(words):
    #             if word.lower() in ["nó", "cái", "này", "chúng", "là", "các", "có", "kích", "thước"]:
    #                 if entities:
    #                     words[i] = list(entities.keys())[-1]
    #             elif word.lower() in self.furniture_keywords:
    #                 entities[word] = sentence
    #         resolved_text.append(" ".join(words))

    #     return " ".join(resolved_text)


    
    def extract_drawing_parameters(self, text):
        """
        Trích xuất thông tin để tạo bản vẽ 2D từ dữ liệu người dùng.
        Đồng thời chuẩn hóa tất cả đơn vị về centimet.
        """
        # Khởi tạo thông số mặc định
        parameters = self.default_parameters.copy()

        # Tiền xử lý văn bản
        clean_text = self.preprocess_text(text)
        print(f"clean_text: {clean_text}")

        # Nhận diện các thực thể
        entities = self.extract_entities(clean_text).to_dict(orient="records")
        print("entities: ", entities)

        # Trích xuất thông tin từ các thực thể
        for entity in entities:
            entity_type = entity.get("type", "").lower()
            value = entity.get("value", "").strip()

            if entity_type in self.furniture_keywords:
                # Phân tích kích thước và đơn vị từ value
                size, unit = self.parse_size_and_unit(value)

                # Lưu vào thông số tương ứng
                if entity_type == "phòng":
                    parameters["room"] = {"type": entity_type, "size": size, "unit": unit}
                elif entity_type == "bàn":
                    parameters["table"] = {"type": entity_type, "size": size, "unit": unit}
                elif entity_type == "ghế":
                    parameters["chair"] = {"type": entity_type, "size": size, "unit": unit}
                elif entity_type == "khoảng cách":
                    parameters["distance_between_tables"] = {"type": entity_type, "size": size, "unit": unit}
                elif entity_type == "lối":
                    parameters["aisle_distance"] = {"type": "lối đi", "size": size, "unit": unit}
                elif entity_type == "quầy":
                    parameters["reception_desk"] = {"type": entity_type, "size": size, "unit": unit}

        # Chuẩn hóa tất cả đơn vị về centimet và xử lý định dạng
        for key, param in parameters.items():
            if param.get("size") and param.get("unit"):
                # Chuyển đổi kích thước về cm
                if "x" in param["size"]:
                    # Nếu kích thước dạng "AxB"
                    dimensions = param["size"].split("x")
                    dimensions_in_cm = []
                    for dim in dimensions:
                        if param["unit"] == "m":
                            dim_in_cm = float(dim) * 100  # Chuyển đổi mét sang cm
                        elif param["unit"] == "cm":
                            dim_in_cm = float(dim)  # Giữ nguyên

                        # Chuyển về dạng số nguyên nếu không có phần thập phân
                        if dim_in_cm.is_integer():
                            dim_in_cm = int(dim_in_cm)
                        dimensions_in_cm.append(str(dim_in_cm))
                    param["size"] = "x".join(dimensions_in_cm)
                else:
                    # Nếu kích thước chỉ có một số
                    if param["unit"] == "m":
                        size_in_cm = float(param["size"]) * 100  # Chuyển đổi mét sang cm
                    elif param["unit"] == "cm":
                        size_in_cm = float(param["size"])  # Giữ nguyên

                    # Chuyển về dạng số nguyên nếu không có phần thập phân
                    if size_in_cm.is_integer():
                        size_in_cm = int(size_in_cm)
                    param["size"] = str(size_in_cm)

                # Cập nhật đơn vị về "cm"
                param["unit"] = "cm"

        print("Trích xuất thông tin (sau chuẩn hóa):", parameters)
        return parameters


    def parse_size_and_unit(self, value):
        """
        Phân tích kích thước và đơn vị từ chuỗi giá trị.
        Ví dụ: "2x3m" -> ("2x3", "m")
        """
        size = ""
        unit = "m"  # Mặc định là mét

        # Tìm kích thước (chữ số và "x")
        for part in value.split():
            if any(char.isdigit() for char in part) or "x" in part:
                size = part.strip()

            # Tìm đơn vị
            if "cm" in part:
                unit = "cm"
            elif "m" in part:
                unit = "m"

        return size, unit

    
    
    def update_parameters(self, user_input):
        """
        Cập nhật thông số từ dữ liệu người dùng.
        Chỉ cập nhật thực thể được đề cập, bảo toàn các giá trị hiện có.
        """
        # Trích xuất thông tin từ đầu vào người dùng
        extracted_params = self.extract_drawing_parameters(user_input)
        updated_entities = []
        invalid_entities = []

        # Cập nhật thông số dựa trên input người dùng
        for key in self.user_parameters:
            # Kiểm tra nếu thực thể có trong dữ liệu người dùng
            if extracted_params.get(key, {}).get("size"):
                size = extracted_params[key]["size"]
                unit = extracted_params[key]["unit"]

                # Kiểm tra tính hợp lệ của kích thước và đơn vị
                if self.is_valid_size(size) and self.is_valid_unit(unit):
                    # Cập nhật giá trị hợp lệ
                    self.user_parameters[key]["size"] = size
                    self.user_parameters[key]["unit"] = unit
                    updated_entities.append(self.user_parameters[key]["type"])
                else:
                    invalid_entities.append(self.user_parameters[key]["type"])

            # Nếu quầy lễ tân được đề cập, cập nhật trạng thái "Có"
            if key == "reception_desk" and extracted_params.get(key, {}).get("size"):
                self.user_parameters[key]["present"] = "Có"

        # Thêm thông số mặc định cho các thực thể còn thiếu
        missing_params = []
        for key, value in self.default_parameters.items():
            if not self.user_parameters[key]["size"]:
                self.user_parameters[key] = value
                missing_params.append(value["type"])
        
        
        # Trả lại thông báo về trạng thái cập nhật
        response = ""
        if updated_entities:
            updated_entities =  self.refill(updated_entities)
            response += f"Các thông số đã được cập nhật: {updated_entities}\n"
        if invalid_entities:
            invalid_entities =  self.refill(invalid_entities)
            response += f"Các thông số không hợp lệ (bỏ qua): {invalid_entities}\n"
        if missing_params:
            missing_params =  self.refill(missing_params)
            response += f"Các thông số còn thiếu đã được tự động khởi tạo: {missing_params}\n"

        return response

    def is_valid_size(self, size):
        """
        Kiểm tra kích thước có hợp lệ không (ví dụ: "700x500", hoặc 12 ).
        """
        if any(char.isdigit() for char in size) or "x" in size:
            dimensions = size.split("x")
            return all(d.strip().isdigit() for d in dimensions)
        
        return False

    def is_valid_unit(self, unit):
        """
        Kiểm tra đơn vị có hợp lệ không (ví dụ: "cm", "m").
        """
        return unit in ["cm", "m"]


    def respond_to_user(self, user_input):
        """
        Phản hồi người dùng, tự động bổ sung dữ liệu thiếu và thông báo.
        """
        # Cập nhật thông số từ dữ liệu người dùng
        missing_params = self.update_parameters(user_input)

        # Chuyển thông số thành DataFrame
        df = pd.DataFrame([
            {
                "Thực thể": param["type"],
                "Kích thước": param["size"],
                "Đơn vị": param.get("unit", ""),
                "Có/Không": param.get("present", "")  # Dành cho quầy lễ tân
            }
            for param in self.user_parameters.values()
        ])

        # Phản hồi người dùng
        if missing_params:
            response = (
                f"\n {missing_params}.\n\n"
                "\nBạn có muốn chỉnh sửa thông số nào không?\n"
            )
        else:
            response = f"Thông số của bạn đã đầy đủ.\n"

        response += "Dưới đây là bảng thông số hiện tại:\n"

        return response

    def handle_user_update(self, user_input):
        """
        Xử lý khi người dùng muốn chỉnh sửa thông số.
        """
        extracted_params = self.extract_drawing_parameters(user_input)
        updated_entities = []

        # Cập nhật thông số dựa trên input người dùng
        for key, param in self.user_parameters.items():
            if key in extracted_params and extracted_params[key].get("size"):
                self.user_parameters[key]["size"] = extracted_params[key]["size"]
                self.user_parameters[key]["unit"] = extracted_params[key]["unit"]
                updated_entities.append(param["type"])
        # Tạo lại bảng thông số sau khi cập nhật
        df = pd.DataFrame([
            {
                "Thực thể": param["type"],
                "Kích thước": param["size"],
                "Đơn vị": param.get("unit", ""),
                "Có/Không": param.get("present", "")  # Dành cho quầy lễ tân
            }
            for param in self.user_parameters.values()
        ])

        response = f"Tôi đã cập nhật thông số các thông số \n"
        response += "\nDưới đây là bảng thông số hiện tại:\n"

        return response
    
    
    def remove_reception_desk(self):
        """
        Loại bỏ quầy lễ tân khỏi thông số hiện tại.
        """
        # Kiểm tra và cập nhật thông số quầy lễ tân
        if "reception_desk" in self.user_parameters:
            self.user_parameters["reception_desk"]["present"] = "Không"
            self.user_parameters["reception_desk"]["size"] = "0x0"  # Đặt kích thước về 0
            self.user_parameters["reception_desk"]["unit"] = ""   # Xóa đơn vị

        # Tạo phản hồi cho người dùng
        response = "Quầy lễ tân đã được loại bỏ khỏi thông số hiện tại."
        return response

    
    
    def refill(self, words):
        """
        Chỉnh sửa danh sách từ theo các quy tắc đã cho.
        """
        # Tạo danh sách mới để lưu kết quả
        updated_words = []
        
        for word in words:
            if word == 'lối':
                updated_words.append('lối đi')
            elif word == 'giữa':
                updated_words.append('khoảng cách giữa các bàn')
            elif word == 'quầy':
                updated_words.append('quầy lễ tân')
            else:
                updated_words.append(word)  # Giữ nguyên nếu không cần thay đổi

        return updated_words
