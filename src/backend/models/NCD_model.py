import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math

class NetCafeModel(nn.Module):
    def __init__(self, parameters):
        """
        PyTorch model for calculating layout of Net Cafe.
        :param parameters: A dictionary containing room, table, chair, and other dimensions.
        """
        super(NetCafeModel, self).__init__()
        self.parameters = parameters
        print("Initialized NetCafeModel with parameters:", self.parameters)  # Debug thông số đầu vào

    def _parse_size(self, size_str):
        """
        Parse size string (e.g., '120x60') into (width, height).
        """
        try:
            dimensions = size_str.split("x")
            parsed_size = float(dimensions[0]), float(dimensions[1])
            print(f"Parsed size from '{size_str}' to {parsed_size}")  # Debug kích thước đã phân tích
            return parsed_size
        except Exception as e:
            print(f"Error parsing size '{size_str}': {e}")
            raise

    def forward(self):
        """
        Tính toán bố trí phòng dựa trên thông số đầu vào.
        :return: Tensor chứa tọa độ của tất cả các thực thể trong phòng.
        """
        print("Bắt đầu tính toán bố trí...")  # Debug: Bắt đầu xử lý
        
        try:
            # Phân tích kích thước phòng
            room_w, room_h = self._parse_size(self.parameters["room"]["size"])
            print(f"Kích thước phòng: chiều rộng={room_w}, chiều cao={room_h}")
            
            # Phân tích kích thước bàn
            table_w, table_h = self._parse_size(self.parameters["table"]["size"])
            print(f"Kích thước bàn: chiều rộng={table_w}, chiều cao={table_h}")
            
            # Phân tích kích thước ghế
            chair_w, chair_h = self._parse_size(self.parameters["chair"]["size"])
            print(f"Kích thước ghế: chiều rộng={chair_w}, chiều cao={chair_h}")
            
            # Kiểm tra trạng thái quầy lễ tân
            if self.parameters["reception_desk"]["present"] == "Có":
                desk_w, desk_h = self._parse_size(self.parameters["reception_desk"]["size"])
                print(f"Kích thước quầy lễ tân: chiều rộng={desk_w}, chiều cao={desk_h}")
            else:
                desk_w, desk_h = 0, 0  # Không có quầy lễ tân
                print("Không có quầy lễ tân trong phòng.")

            # Khoảng cách giữa các bàn và lối đi
            distance_bt = float(self.parameters["distance_between_tables"]["size"])
            aisle_dist = float(self.parameters["aisle_distance"]["size"])
            print(f"Khoảng cách giữa các bàn: {distance_bt}, Khoảng cách lối đi: {aisle_dist}")

            # Khởi tạo cấu trúc lưu trữ bố trí
            layout = {
                "room": {"width": room_w, "height": room_h},
                "tables": [],
                "chairs": [],
                "reception_desk": None if desk_w == 0 else {"x": 0, "y": room_h - desk_h, "width": desk_w, "height": desk_h}
            }

            # Nếu có quầy lễ tân, khởi tạo reception_tensor
            reception_tensor = None if desk_w == 0 else torch.tensor(
                [[layout["reception_desk"]["x"], layout["reception_desk"]["y"], desk_w, desk_h]]
            )

            # Tạo vị trí bàn và ghế
            table_positions, chair_positions = self.generate_table_positions_and_chairs(
                room_w, room_h, table_w, table_h, chair_w, chair_h, desk_w, desk_h, aisle_dist, distance_bt
            )
            
            print(f"Vị trí bàn được tính: {table_positions}")
            print(f"Vị trí ghế được tính: {chair_positions}")

            # Chuyển đổi sang tensor
            table_tensor = torch.tensor(table_positions)
            chair_tensor = torch.tensor(chair_positions)

            print(f"Tensor cuối cùng:\nBàn: {table_tensor}\nGhế: {chair_tensor}\nQuầy lễ tân: {reception_tensor}")
            return table_tensor, chair_tensor, reception_tensor
        except Exception as e:
            print(f"Lỗi trong quá trình tính toán: {e}")
            raise

    def generate_table_positions_and_chairs(self, room_w, room_h, table_w, table_h, chair_w, chair_h, desk_w, desk_h, aisle_dist, gap):
        """
        Tạo vị trí bàn và ghế dựa trên khung bounding box.
        """
        # Tạo khung bounding box cho từng bàn và ghế
        bounding_box = self.create_bounding_boxes(table_w, table_h, chair_h, gap)

        # Tối ưu hóa việc đặt các khung trong phòng
        bounding_boxes = self.optimize_bounding_boxes(room_w, room_h, bounding_box, desk_w, desk_h, aisle_dist, gap)

        # Từ các bounding box, vẽ bàn và ghế
        table_positions, chair_positions = self.place_entities_from_boxes(bounding_boxes, table_w, table_h, chair_w, chair_h)
        return table_positions, chair_positions

    def create_bounding_boxes(self, table_w, table_h, chair_h, gap):
        """
        Tạo khung bounding box cho bàn và ghế, kèm theo thông tin hướng.
        :param table_w: Chiều rộng của bàn.
        :param table_h: Chiều cao của bàn.
        :param chair_h: Chiều cao của ghế.
        :param gap: Khoảng cách giữa bàn và ghế.
        :return: Bounding box chứa thông tin width, height, và orientation.
        """
        frame_w = table_w
        frame_h = table_h + chair_h + gap
        orientation = 0  # Hướng mặc định là 0 độ
        print(f"Tạo khung bounding box: width={frame_w}, height={frame_h}, orientation={orientation}")
        return {"width": frame_w, "height": frame_h, "orientation": orientation}


    def optimize_bounding_boxes(self, room_w, room_h, bounding_box, desk_w, desk_h, aisle_dist, gap):
        """
        Tối ưu hóa việc đặt các khung bounding box vào phòng, đảm bảo đổ từ tường bên phải về bên trái
        và hỗ trợ tạo dãy back-to-back nếu không đủ chỗ.
        """
        bounding_boxes = []
        box_width, box_height, box_orientation = bounding_box["width"], bounding_box["height"], bounding_box["orientation"]

        # Bắt đầu từ góc dưới bên phải
        current_x, current_y = room_w - box_width, room_h - box_height
        rows = []  # Lưu trữ các dãy
        back_to_back = False  # Cờ để kiểm tra dãy back-to-back

        while current_y >= 0:
            current_row = []
            while current_x >= 0:
                # Kiểm tra va chạm với quầy lễ tân
                if desk_w > 0 and current_x < desk_w and current_y + box_height > room_h - desk_h:
                    current_x -= box_width + aisle_dist  # Lùi lại, tránh va chạm
                    continue

                # Điều chỉnh tọa độ sát tường nếu cần
                adjusted_x = max(0, current_x)
                adjusted_y = max(0, current_y)

                # Thêm bounding box hợp lệ
                box = {
                    "x": adjusted_x,
                    "y": adjusted_y,
                    "width": box_width,
                    "height": box_height,
                    "orientation": box_orientation  # Lưu hướng hiện tại
                }
                bounding_boxes.append(box)
                current_row.append(box)

                # Lùi sang bên trái với khoảng cách gap
                current_x -= box_width + gap

            rows.append(current_row)  # Lưu dãy hiện tại

            # # Kiểm tra nếu không đủ chỗ để tạo dãy tiếp theo và đủ chỗ cho aisle_dist
            # if current_y - box_height - aisle_dist < 0 and not back_to_back and current_y - box_height - gap >= 0:
            #     # Đặt dãy back-to-back
            #     back_to_back = True
            #     current_y += box_height + gap 
            #     #box_orientation = (box_orientation + 180) % 360  # Xoay 180 độ

            #     # Xoay dãy trước đó
            #     for box in rows[-1]:
            #         rotated_box = self.rotate_bounding_box(box, 180, room_w, room_h, aisle_dist)
            #         bounding_boxes[bounding_boxes.index(box)] = rotated_box  # Cập nhật
            #         # current_y = rotated_box["y"] + rotated_box["height"] + aisle_dist
                
            # else:
                # Chuyển sang hàng tiếp theo (lên trên) với khoảng cách aisle_dist
            current_x = room_w - box_width
            current_y -= box_height + aisle_dist
            back_to_back = False  # Reset cờ

        print(f"Bounding boxes tối ưu theo khoảng cách gap và aisle_dist (từ phải sang trái): {bounding_boxes}")
        return bounding_boxes


    def place_entities_from_boxes(self, bounding_boxes, table_w, table_h, chair_w, chair_h, gap=5):
        """
        Tạo vị trí bàn và ghế từ danh sách bounding box, dựa trên hướng của bàn.
        :param bounding_boxes: Danh sách bounding box (bao gồm hướng orientation).
        :param table_w: Chiều rộng của bàn.
        :param table_h: Chiều cao của bàn.
        :param chair_w: Chiều rộng của ghế.
        :param chair_h: Chiều cao của ghế.
        :param gap: Khoảng cách giữa bàn và ghế.
        :return: Danh sách vị trí bàn và ghế.
        """
        table_positions = []
        chair_positions = []

        for box in bounding_boxes:
            # Đặt bàn trong bounding box
            table_x = box["x"]
            table_y = box["y"]
            orientation = box["orientation"]  # Hướng mặc định là 0 độ

            # Tùy theo hướng (orientation), cập nhật vị trí bàn
            if orientation == 0:  # Bàn trên, ghế dưới
                table_positions.append([table_x, table_y + box["height"] - table_h, table_w, table_h])
                chair_x = table_x + (table_w - chair_w) / 2
                chair_y = table_y + box["height"] - table_h - chair_h - gap

            elif orientation == 90:  # Bàn bên trái, ghế bên phải
                table_positions.append([table_x, table_y, table_h, table_w])  # Xoay bàn 90 độ
                chair_x = table_x + table_h + gap
                chair_y = table_y + (table_w - chair_h) / 2

            elif orientation == 180:  # Bàn dưới, ghế trên
                table_positions.append([table_x, table_y, table_w, table_h])
                chair_x = table_x + (table_w - chair_w) / 2
                chair_y = table_y + table_h + gap

            elif orientation == 270:  # Bàn bên phải, ghế bên trái
                table_positions.append([table_x + box["width"] - table_h, table_y, table_h, table_w])  # Xoay bàn 270 độ
                chair_x = table_x - chair_w - gap
                chair_y = table_y + (table_w - chair_h) / 2

            else:
                raise ValueError(f"Hướng không hợp lệ: {orientation}")

            # Thêm ghế vào danh sách nếu không vượt ra ngoài bounding box
            if box["x"] <= chair_x <= box["x"] + box["width"] - chair_w and \
                    box["y"] <= chair_y <= box["y"] + box["height"] - chair_h:
                chair_positions.append([chair_x, chair_y, chair_w, chair_h])
            else:
                print(f"Ghế tại ({chair_x}, {chair_y}) vượt ra ngoài bounding box, bỏ qua.")

        print(f"Bàn được đặt: {table_positions}")
        print(f"Ghế được đặt: {chair_positions}")
        return table_positions, chair_positions



    def rotate_bounding_box(self, box, angle, room_w, room_h, aisle_dist):
        """
        Xoay bounding box theo góc chỉ định (90, 180 hoặc 270 độ) và cập nhật hướng.

        :param box: Từ điển chứa thông tin của bounding box {'x', 'y', 'width', 'height', 'orientation'}.
        :param angle: Góc xoay (90, 180 hoặc 270 độ).
        :param room_w: Chiều rộng phòng.
        :param room_h: Chiều cao phòng.
        :return: Bounding box mới sau khi xoay và điều chỉnh.
        """
        if angle not in [90, 180, 270]:
            raise ValueError("Góc xoay chỉ có thể là 90, 180 hoặc 270 độ.")
        
        x, y, width, height = box['x'], box['y'], box['width'], box['height']
        # Nếu 'orientation' không tồn tại trong box, mặc định là 0
        if 'orientation' in box:
            orientation = box['orientation']
        else:
            orientation = 0

        if angle == 90:
            # Hoán đổi width và height khi xoay 90 độ
            new_width, new_height = height, width
            
            # Tính vị trí mới
            new_x = y
            new_y = room_w - x - new_width

        elif angle == 180:
            # Không thay đổi width và height khi xoay 180 độ
            new_width, new_height = width, height
            
            # Tính vị trí mới
            new_x = x#room_w - x - new_width
            new_y = room_h - y - 50
            print(f"x", y, new_y)

        elif angle == 270:
            # Hoán đổi width và height khi xoay 270 độ
            new_width, new_height = height, width
            
            # Tính vị trí mới
            new_x = room_h - y - new_height
            new_y = x

        # Điều chỉnh bounding box nếu vượt ra ngoài giới hạn phòng
        if new_x < 0:
            new_x = 0
        if new_y < 0:
            new_y = 0
        if new_x + new_width > room_w:
            new_x = room_w - new_width
        if new_y + new_height > room_h:
            new_y = room_h - new_height

        # Cập nhật hướng của bàn sau khi xoay
        new_orientation = (orientation + angle) % 360

        return {
            "x": new_x,
            "y": new_y,
            "width": new_width,
            "height": new_height,
            "orientation": new_orientation  # Lưu hướng mới của bàn
        }


    # def get_table_orientation(table):
    #     """
    #     Lấy hướng của bàn.
    #     :param table: Dictionary chứa thông tin bàn {'x', 'y', 'width', 'height', 'orientation'}.
    #     :return: String mô tả hướng của bàn.
    #     """
    #     orientation_map = {
    #         0: "Hướng lên",
    #         90: "Hướng sang phải",
    #         180: "Hướng xuống",
    #         270: "Hướng sang trái"
    #     }
    #     return orientation_map.get(table['orientation'], "Không xác định")

    
    # def generate_table_positions(self, room_w, room_h, table_w, table_h, desk_w, desk_h, aisle_dist, distance_bt):
    #     """
    #     Tạo vị trí cho các bàn và lưu các ô đã chiếm dụng.
    #     """
    #     table_positions = []
    #     occupied_positions = []

    #     start_x = desk_w + aisle_dist
    #     start_y = room_h - desk_h - aisle_dist

    #     num_tables_per_row = int((room_w - start_x) // (table_w + aisle_dist))
    #     num_rows = int((start_y) // (table_h + distance_bt))

    #     for row in range(num_rows):
    #         for col in range(num_tables_per_row):
    #             table_x = start_x + col * (table_w + aisle_dist)
    #             table_y = start_y - row * (table_h + distance_bt)

    #             # Kiểm tra va chạm với các thực thể đã được chiếm dụng
    #             if self.is_collision([table_x, table_y, table_w, table_h], room_w, room_h, None, occupied_positions):
    #                 print(f"Bàn tại ({table_x}, {table_y}) bị chồng lấn, bỏ qua.")
    #                 continue

    #             # Thêm bàn vào danh sách
    #             table_positions.append([table_x, table_y, table_w, table_h])
    #             occupied_positions.append([table_x, table_y, table_w, table_h])  # Lưu ô đã chiếm dụng

    #     print(f"Vị trí bàn: {table_positions}")
    #     return table_positions, occupied_positions



    # def generate_chair_positions(self, table_positions, chair_w, chair_h, aisle_dist, room_w, room_h, occupied_positions):
    #     """
    #     Tạo vị trí cho ghế dựa trên vị trí bàn và lưu các ô đã chiếm dụng.
    #     """
    #     chair_positions = []
    #     for table in table_positions:
    #         table_x, table_y, table_w, table_h = table

    #         # Ưu tiên đặt ghế trên hoặc dưới theo chiều dài của bàn
    #         if table_w >= table_h:
    #             # Ghế trên
    #             chair_top = [table_x + (table_w - chair_w) / 2, table_y - chair_h - aisle_dist, chair_w, chair_h]
    #             # Ghế dưới
    #             chair_bottom = [table_x + (table_w - chair_w) / 2, table_y + table_h + aisle_dist, chair_w, chair_h]

    #             if not self.is_collision(chair_top, room_w, room_h, None, occupied_positions):
    #                 chair_positions.append(chair_top)
    #                 occupied_positions.append(chair_top)
    #             elif not self.is_collision(chair_bottom, room_w, room_h, None, occupied_positions):
    #                 chair_positions.append(chair_bottom)
    #                 occupied_positions.append(chair_bottom)
    #         else:
    #             # Ghế bên trái
    #             chair_left = [table_x - chair_w - aisle_dist, table_y + (table_h - chair_h) / 2, chair_w, chair_h]
    #             # Ghế bên phải
    #             chair_right = [table_x + table_w + aisle_dist, table_y + (table_h - chair_h) / 2, chair_w, chair_h]

    #             if not self.is_collision(chair_left, room_w, room_h, None, occupied_positions):
    #                 chair_positions.append(chair_left)
    #                 occupied_positions.append(chair_left)
    #             elif not self.is_collision(chair_right, room_w, room_h, None, occupied_positions):
    #                 chair_positions.append(chair_right)
    #                 occupied_positions.append(chair_right)

    #     print(f"Vị trí ghế được tạo: {chair_positions}")
    #     return chair_positions



        
        
    # def is_collision(self, entity, room_w, room_h, reception_positions, occupied_positions):
    #     """
    #     Kiểm tra xem thực thể có va chạm với quầy lễ tân, bàn, ghế hoặc vượt ra ngoài phòng không.
    #     """
    #     x, y, w, h = entity

    #     # Kiểm tra nếu vượt ra ngoài phòng
    #     if x < 0 or y < 0 or x + w > room_w or y + h > room_h:
    #         return True

    #     # Kiểm tra va chạm với quầy lễ tân (nếu có)
    #     if reception_positions is not None:
    #         for reception in reception_positions:
    #             rx, ry, rw, rh = reception
    #             if not (x + w <= rx or x >= rx + rw or y + h <= ry or y >= ry + rh):
    #                 return True

    #     # Kiểm tra va chạm với các thực thể đã được đặt (bàn và ghế)
    #     for occupied in occupied_positions:
    #         ox, oy, ow, oh = occupied
    #         if not (x + w <= ox or x >= ox + ow or y + h <= oy or y >= oy + oh):
    #             return True

    #     return False




        # def _is_position_valid(self, chair, reception_positions, room_w, room_h):
        #     """
        #     Kiểm tra xem vị trí ghế có hợp lệ không (không va chạm với quầy lễ tân và trong phạm vi phòng).
        #     """
        #     chair_x, chair_y, chair_w, chair_h = chair

        #     # Kiểm tra ghế nằm trong phòng
        #     if chair_x < 0 or chair_y < 0 or (chair_x + chair_w) > room_w or (chair_y + chair_h) > room_h:
        #         return False

        #     # Kiểm tra va chạm với quầy lễ tân
        #     if reception_positions is not None:
        #         for reception in reception_positions:
        #             reception_x, reception_y, reception_w, reception_h = reception
        #             if not (
        #                 chair_x + chair_w < reception_x or  # Trái hoàn toàn
        #                 chair_x > reception_x + reception_w or  # Phải hoàn toàn
        #                 chair_y + chair_h < reception_y or  # Trên hoàn toàn
        #                 chair_y > reception_y + reception_h  # Dưới hoàn toàn
        #             ):
        #                 return False

        #     return True


            # def calculate_starting_positions(self, desk_w, desk_h, room_w, room_h, aisle_dist):
    #     """
    #     Tính toán vị trí bắt đầu cho các thực thể dựa trên quầy lễ tân.
    #     """
    #     start_x = desk_w + aisle_dist if desk_w > 0 else aisle_dist
    #     start_y = room_h - desk_h - aisle_dist if desk_h > 0 else room_h - aisle_dist
    #     print(f"Vị trí bắt đầu: start_x={start_x}, start_y={start_y}")
    #     return start_x, start_y


    # def calculate_table_grid(self, start_x, start_y, room_w, room_h, table_w, table_h, aisle_dist, distance_bt):
    #     """
    #     Tính toán số lượng bàn trên mỗi hàng và số hàng.
    #     """
    #     num_tables_per_row = int((room_w - start_x) // (table_w + aisle_dist))
    #     num_rows = int((start_y) // (table_h + distance_bt))
    #     print(f"Số bàn trên mỗi hàng: {num_tables_per_row}, Số hàng: {num_rows}")
    #     return num_tables_per_row, num_rows






