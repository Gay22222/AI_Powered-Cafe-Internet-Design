import os
import matplotlib
matplotlib.use('Agg')  # Đảm bảo sử dụng backend không GUI
from flask import Flask, render_template, request, jsonify, send_file
import uuid
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.nlp_model import NLPModel
from models.NCD_model import NetCafeModel 

app = Flask(
    __name__,
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/public')),
    template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/templates'))
)

# Khởi tạo NLPModel
nlp_model = NLPModel()

# Tạo folder để lưu file thiết kế
DESIGN_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/public/designs"))
if not os.path.exists(DESIGN_FOLDER):
    os.makedirs(DESIGN_FOLDER)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    if user_input.lower() == "xác nhận":
        try:
            session_id = str(uuid.uuid4())
            file_url = draw_layout(session_id)  # Hàm này trả về URL của hình ảnh đã tạo
            return send_file(
                os.path.join(DESIGN_FOLDER, f"design_{session_id}.png"),
                mimetype="application/octet-stream",
                as_attachment=True,
                download_name=f"design_{session_id}.png"
            )
        except Exception as e:
            return jsonify({"error": f"Failed to generate design: {str(e)}"}), 500

    if "tôi muốn" in user_input.lower() or "cập nhật" in user_input.lower():
        response = nlp_model.handle_user_update(user_input)
        return jsonify({
            "response": response,
            "parameters": nlp_model.user_parameters
        })
    # Khi người dùng yêu cầu loại bỏ quầy lễ tân
    if "loại bỏ quầy lễ tân" in user_input.lower():
        response = nlp_model.remove_reception_desk()
        return jsonify({
            "response": response,
            "parameters": nlp_model.user_parameters
        })
        
    response = nlp_model.respond_to_user(user_input)
    return jsonify({
        "response": response,
        "parameters": nlp_model.user_parameters
    })
    


def draw_layout(session_id):

    try:
        net_cafe_model = NetCafeModel(nlp_model.user_parameters)
    except Exception as e:
        print(f"Error initializing NetCafeModel: {e}")
        raise

    # Tính toán layout tensors
    try:
        table_tensor, chair_tensor, reception_tensor = net_cafe_model.forward()
        print(f"Layout tensors: Tables - {table_tensor}, Chairs - {chair_tensor}, Reception - {reception_tensor}")
    except Exception as e:
        print(f"Error calculating layout tensors: {e}")
        raise

    # Tạo hình ảnh matplotlib
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        print("Matplotlib figure and axis created.")
    except Exception as e:
        print(f"Error creating Matplotlib figure: {e}")
        raise

    # Lấy kích thước phòng
    try:
        room_width, room_height = net_cafe_model._parse_size(net_cafe_model.parameters["room"]["size"])
        print(f"Parsed room dimensions: width={room_width}, height={room_height}")
    except Exception as e:
        print(f"Error parsing room dimensions: {e}")
        raise

    # Vẽ phòng
    try:
        ax.set_xlim(0, room_width)
        ax.set_ylim(0, room_height)
        ax.add_patch(patches.Rectangle((0, 0), room_width, room_height, edgecolor='black', fill=None))
        # ax.invert_yaxis()
        print("Room drawn successfully.")
    except Exception as e:
        print(f"Error drawing room: {e}")
        raise

    # Vẽ quầy lễ tân nếu có
    try:
        if net_cafe_model.parameters["reception_desk"]["present"] == "Có":
            reception = reception_tensor[0].numpy()
            ax.add_patch(patches.Rectangle(
                (reception[0], reception[1]),
                reception[2],
                reception[3],
                edgecolor='blue',
                facecolor='lightblue',
                label="Reception Desk"
            ))
            # Thêm nhãn cho quầy lễ tân
            ax.text(
                reception[0] + reception[2] / 2,
                reception[1] + reception[3] / 2,
                "Quầy lễ tân",
                color="black",
                fontsize=8,
                ha="center",
                va="center"
            )
            print(f"Reception desk drawn at: {reception}.")
    except Exception as e:
        print(f"Error drawing reception desk: {e}")
        raise

    # Vẽ các bàn
    try:
        for table in table_tensor.numpy():
            ax.add_patch(patches.Rectangle(
                (table[0], table[1]),
                table[2],
                table[3],
                edgecolor='green',
                facecolor='lightgreen',
                label="Table"
            ))
            # Thêm nhãn cho bàn
            ax.text(
                table[0] + table[2] / 2,
                table[1] + table[3] / 2,
                "Bàn",
                color="black",
                fontsize=8,
                ha="center",
                va="center"
            )
        print("Tables drawn successfully.")
    except Exception as e:
        print(f"Error drawing tables: {e}")
        raise

    # Vẽ các ghế
    try:
        for chair in chair_tensor.numpy():
            ax.add_patch(patches.Rectangle(
                (chair[0], chair[1]),
                chair[2],
                chair[3],
                edgecolor='orange',
                facecolor='yellow',
                label="Chair"
            ))
            # Thêm nhãn cho ghế
            ax.text(
                chair[0] + chair[2] / 2,
                chair[1] + chair[3] / 2,
                "Ghế",
                color="black",
                fontsize=6,
                ha="center",
                va="center"
            )
        print("Chairs drawn successfully.")
    except Exception as e:
        print(f"Error drawing chairs: {e}")
        raise

    # # Thêm các tham chiếu kích thước
    # try:
    #     # Chiều dài phòng (horizontal reference)
    #     ax.annotate(
    #         f"{room_width} cm",
    #         xy=(room_width / 2, -10), xycoords='data',
    #         fontsize=10, ha='center', va='center',
    #         annotation_clip=False
    #     )
    #     # Chiều rộng phòng (vertical reference)
    #     ax.annotate(
    #         f"{room_height} cm",
    #         xy=(-10, room_height / 2), xycoords='data',
    #         fontsize=10, ha='center', va='center',
    #         rotation=90,
    #         annotation_clip=False
    #     )
    #     print("Dimension references added.")
    # except Exception as e:
    #     print(f"Error adding dimension references: {e}")
    #     raise

    # Thêm nhãn cho trục tọa độ
    try:
        ax.set_xlabel("Chiều dài (cm)", fontsize=12)
        ax.set_ylabel("Chiều rộng (cm)", fontsize=12)
    except Exception as e:
        print(f"Error adding axis labels: {e}")
        raise

    # Hoàn tất và lưu hình ảnh
    try:
        ax.set_title("Net Cafe Layout")
        #ax.invert_yaxis()
        output_file = os.path.join(DESIGN_FOLDER, f"design_{session_id}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Layout saved to: {output_file}")
    except Exception as e:
        print(f"Error saving layout: {e}")
        raise

    # Xóa file sau 10 phút
    try:
        threading.Thread(target=delete_file_after_timeout, args=(output_file, 600)).start()
        print("Thread to delete file after timeout started.")
    except Exception as e:
        print(f"Error starting delete thread: {e}")
        raise

    # Trả về URL
    return f"/static/designs/design_{session_id}.png"



def delete_file_after_timeout(file_path, timeout):
    import time
    time.sleep(timeout)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete file {file_path}: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
