// Đợi DOM được tải hoàn toàn
document.addEventListener("DOMContentLoaded", () => {
    // Các phần tử HTML
    const chatForm = document.getElementById("chatForm");
    const userMessageInput = document.getElementById("userMessage");
    const chatOutput = document.getElementById("chatOutput");

    // Hàm thêm tin nhắn vào giao diện
    function appendMessage(content, sender = "bot", isTyping = false, isError = false) {
        const messageElement = document.createElement("p");
        messageElement.classList.add(`${sender}-message`);
        if (isTyping) {
            messageElement.textContent = "Đang tạo...";
            messageElement.classList.add("typing");
        } else if (isError) {
            messageElement.textContent = content;
            messageElement.classList.add("error-message");
        } else {
            messageElement.textContent = content;
        }
        chatOutput.appendChild(messageElement);

        // Cuộn xuống để hiển thị tin nhắn mới
        chatOutput.scrollTop = chatOutput.scrollHeight;

        return messageElement;
    }

    // Hàm tạo bảng thông số
    function displayParameterTable(parameters) {
        const table = document.createElement("table");
        table.classList.add("parameters-table");

        // Thêm tiêu đề bảng
        const headerRow = document.createElement("tr");
        ["Thực thể", "Kích thước", "Đơn vị", "Có/Không"].forEach((header) => {
            const th = document.createElement("th");
            th.textContent = header;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        // Thêm dữ liệu vào bảng
        Object.values(parameters).forEach((param) => {
            const row = document.createElement("tr");
            // Xử lý tên thực thể
            let type = param.type;
            if (type === "lối") {
                type = "lối đi";
            } else if (type === "giữa") {
                type = "khoảng cách giữa các bàn";
            } else if (type === "quầy") {
                type = "quầy lễ tân";
             }
            const fields = [
                type,
                param.size,
                param.unit || "",
                param.present || "" // Dành cho quầy lễ tân
            ];
            fields.forEach((field) => {
                const td = document.createElement("td");
                td.textContent = field;
                row.appendChild(td);
            });
            table.appendChild(row);
        });

        // Thêm bảng vào giao diện
        chatOutput.appendChild(table);
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }

    // Gửi tin nhắn đến API Flask
    async function sendMessageToAPI(message) {
        // Hiển thị thông báo "Đang gõ..."
        const typingElement = appendMessage("", "bot", true);
    
        try {
            // Gửi yêu cầu POST đến API
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message }),
            });
            
            // Kiểm tra phản hồi
            if (!response.ok) {
                throw new Error("Có lỗi xảy ra khi kết nối với server.");
            }
    
            // Kiểm tra xem API trả về file hay JSON
            const contentType = response.headers.get("Content-Type");
    
            if (contentType.includes("application/json")) {
                // Xử lý JSON phản hồi
                const data = await response.json();
                chatOutput.removeChild(typingElement);
    
                // Kiểm tra nếu phản hồi là bảng thông số
                if (data.parameters) {
                    appendMessage(data.response, "bot");
                    displayParameterTable(data.parameters);
                    appendMessage(
                        "Nếu bạn thấy thông số đã ổn, hãy nhập 'Xác nhận' để tôi bắt đầu tạo bản vẽ.",
                        "bot"
                    );
                } else if (data.response) {
                    // Hiển thị phản hồi dạng text
                    appendMessage(data.response, "bot");
                } else {
                    // Hiển thị thông báo lỗi nếu không nhận được dữ liệu phù hợp
                    appendMessage("Không nhận được phản hồi phù hợp từ server.", "bot", false, true);
                }
            } else if (contentType.includes("application/octet-stream")) {
                // Tạo URL từ blob để hiển thị hình ảnh
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
    
                // Hiển thị hình ảnh trong giao diện
                displayImageWithDownload(url);
                chatOutput.removeChild(typingElement);
    
            
                // Thông báo thành công
                appendMessage("File thiết kế đã được tạo. ", "bot");
            }
        } catch (error) {
            // Xóa thông báo "Đang gõ..."
            chatOutput.removeChild(typingElement);
    
            // Hiển thị lỗi
            appendMessage("Lỗi: " + error.message, "bot", false, true);
        }
    }
    
    // Hàm hiển thị hình ảnh với nút tải về
    function displayImageWithDownload(imageUrl) {
        const imageContainer = document.createElement("div");
        imageContainer.classList.add("image-container");
    
        // Tạo thẻ hình ảnh
        const imageElement = document.createElement("img");
        imageElement.src = imageUrl;
        imageElement.alt = "Thiết kế phòng net";
        imageElement.classList.add("design-image");
    
        // Tạo nút tải về
        const downloadButton = document.createElement("a");
        downloadButton.href = imageUrl;
        downloadButton.download = "design.png"; // Tên file khi tải về
        downloadButton.textContent = "Tải về";
        downloadButton.classList.add("download-button");
    
        // Thêm hình ảnh và nút tải về vào giao diện
        imageContainer.appendChild(imageElement);
        imageContainer.appendChild(downloadButton);
        chatOutput.appendChild(imageContainer);
    
        // Cuộn xuống để hiển thị
        chatOutput.scrollTop = chatOutput.scrollHeight;
    }
    

    // Xử lý sự kiện gửi tin nhắn
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault(); // Ngăn tải lại trang

        // Lấy tin nhắn từ người dùng
        const userMessage = userMessageInput.value.trim();

        if (userMessage) {
            // Hiển thị tin nhắn của người dùng trên giao diện
            appendMessage(userMessage, "user");

            // Gửi tin nhắn đến API
            sendMessageToAPI(userMessage);

            // Xóa input sau khi gửi
            userMessageInput.value = "";
        }
    });
});
