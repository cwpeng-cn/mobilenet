from utils import plot_loss

loss_file_path = "LogV1.txt"
plot_loss(loss_file_path, title="MobileNet V1 Loss", save_name="V1_loss.png")

loss_file_path = "LogV2.txt"
plot_loss(loss_file_path, title="MobileNet V2 Loss", save_name="V2_loss.png")
