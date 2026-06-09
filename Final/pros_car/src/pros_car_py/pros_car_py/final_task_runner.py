import rclpy
import threading
import time
import json
import math
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose

# 重用現有套件
from pros_car_py.ros_communicator import RosCommunicator
from pros_car_py.data_processor import DataProcessor
from pros_car_py.ik_solver import PybulletRobotController
from pros_car_py.arm_controller_2D import ArmController

class FinalTaskRunner:
    def __init__(self):
        # 實例化 ROS 節點與各控制模組
        self.node = RosCommunicator()
        self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.ros_thread.start()

        self.data_processor = DataProcessor(self.node)
        self.ik_solver = PybulletRobotController(end_eff_index=5)
        self.yolo_sub = self.node.create_subscription(String, '/yolo/object/offset', self._yolo_callback, 10)
        self.yolo_data = []

        # 初始化位姿發布器
        self.initial_pose_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.arm_controller = ArmController(self.node, self.data_processor)

        # 路徑記錄
        self.trajectory = []
        self.is_recording = False
        self.record_thread = threading.Thread(target=self._record_trajectory_loop, daemon=True)

    def _yolo_callback(self, msg):
        try:
            self.yolo_data = json.loads(msg.data)
        except:
            self.yolo_data = []

    def publish_initial_pose(self):
        self.node.get_logger().info('發布 AMCL 初始位姿 (0,0,0) 並等待定位收斂...')
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        for i in range(10):
            msg.header.stamp = self.node.get_clock().now().to_msg()
            self.initial_pose_pub.publish(msg)
            time.sleep(1.0)
            if self.get_current_pose() is not None:
                self.node.get_logger().info('✅ AMCL 定位已確認')
                return
        self.node.get_logger().warn('⚠️ AMCL 似乎沒有回應，初始位姿可能未成功設定')

    def get_current_pose(self):
        """從 AMCL 取得目前位姿，回傳 (x, y, yaw) 或 None"""
        pose_msg = self.node.get_latest_amcl_pose()
        if pose_msg is None:
            return None
        p = pose_msg.pose.pose.position
        o = pose_msg.pose.pose.orientation
        siny_cosp = 2.0 * (o.w * o.z + o.x * o.y)
        cosy_cosp = 1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (p.x, p.y, yaw)

    # ==================================================================
    #  路徑記錄 (Trajectory Recording)
    # ==================================================================

    def start_recording(self):
        self.is_recording = True
        self.record_thread.start()
        self.node.get_logger().info("開始記錄行走路徑 (0.5Hz)...")

    def stop_recording_and_save(self, filename):
        self.is_recording = False
        try:
            with open(filename, 'w') as f:
                json.dump(self.trajectory, f, indent=2)
            self.node.get_logger().info(f"✅ 行走路徑已儲存至 {filename} (共 {len(self.trajectory)} 筆資料)")
        except Exception as e:
            self.node.get_logger().error(f"儲存路徑失敗: {e}")

    def _record_trajectory_loop(self):
        start_time = time.time()
        while rclpy.ok():
            if not self.is_recording:
                time.sleep(0.1)
                continue
            pose = self.get_current_pose()
            if pose:
                x, y, yaw = pose
                t = time.time() - start_time
                self.trajectory.append({
                    "time": round(t, 2),
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "yaw_rad": round(yaw, 4),
                    "yaw_deg": round(math.degrees(yaw), 2)
                })
            time.sleep(0.5)

    # ==================================================================
    #  工具函式
    # ==================================================================

    @staticmethod
    def normalize_angle(angle):
        """將角度歸一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def stop(self):
        """緊急煞車"""
        self.node.publish_raw_car_control([0.0, 0.0, 0.0, 0.0])

    # ==================================================================
    #  PID 原地旋轉
    # ==================================================================

    def rotate_to_yaw(self, target_yaw, tolerance_deg=1.5, timeout=30.0):
        """
        使用 PID 控制器原地旋轉到指定的全局 yaw 角度。
        直接控制輪速，完全不經過 Nav2。
        """
        tolerance = math.radians(tolerance_deg)

        # PID 參數
        Kp = 350.0
        Ki = 80.0  # 大幅提升 Ki，解決卡住時加大力道太遲鈍的問題
        Kd = 50.0

        MAX_ROTATE_SPEED = 400.0
        MIN_ROTATE_SPEED = 220.0

        integral = 0.0
        prev_error = None
        start_time = time.time()
        dt = 0.05
        
        # 紀錄起始位置以抑制旋轉時的往前飄移
        start_pose = self.get_current_pose()
        start_x = start_pose[0] if start_pose else 0.0
        start_y = start_pose[1] if start_pose else 0.0

        on_target_count = 0
        ON_TARGET_REQUIRED = 6

        self.node.get_logger().info(
            f"🔄 開始 PID 原地旋轉: 目標 {math.degrees(target_yaw):.1f}°, 容忍 ±{tolerance_deg}°"
        )

        while rclpy.ok() and (time.time() - start_time) < timeout:
            pose = self.get_current_pose()
            if pose is None:
                time.sleep(dt)
                continue

            current_yaw = pose[2]
            error = self.normalize_angle(target_yaw - current_yaw)

            if abs(error) < tolerance:
                on_target_count += 1
                if on_target_count >= ON_TARGET_REQUIRED:
                    self.stop()
                    self.node.get_logger().info(
                        f"✅ 旋轉完成！目標: {math.degrees(target_yaw):.1f}°, "
                        f"實際: {math.degrees(current_yaw):.1f}°, "
                        f"誤差: {math.degrees(error):.1f}°"
                    )
                    return True
            else:
                on_target_count = 0

            # PID 計算
            integral += error * dt
            # 放寬積分上限，讓卡住時可以持續累積力量推過靜摩擦力
            integral = max(-10.0, min(10.0, integral))

            if prev_error is not None:
                derivative = (error - prev_error) / dt
            else:
                derivative = 0.0
            prev_error = error

            output = Kp * error + Ki * integral + Kd * derivative
            output = max(-MAX_ROTATE_SPEED, min(MAX_ROTATE_SPEED, output))

            # 最小輪速保證（克服靜摩擦力）
            # 只要還沒進入容忍範圍，就確保輪速至少達到最低值
            # 為了避免方向判斷錯誤，如果 output 極小但 error 是正的，我們強制賦予正確符號
            if abs(error) > tolerance:
                if abs(output) < MIN_ROTATE_SPEED:
                    output = MIN_ROTATE_SPEED if error > 0 else -MIN_ROTATE_SPEED

            # 計算防飄移補償
            v_drift_comp = 0.0
            if start_pose:
                dx = pose[0] - start_x
                dy = pose[1] - start_y
                # 投影到當前車頭方向，得知往前飄移了多少公尺
                forward_drift = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
                # 簡單的 P 控制器把車子往後拉 (如果前飄)
                drift_Kp = 600.0
                v_drift_comp = -drift_Kp * forward_drift
                # 限制補償力道，避免過度干擾旋轉
                v_drift_comp = max(-150.0, min(150.0, v_drift_comp))

            # 差速：左輪 = -output, 右輪 = +output（正 output = 逆時針左轉）
            # 加上 v_drift_comp 補償往前飄移
            v_left = -output + v_drift_comp
            v_right = output + v_drift_comp
            self.node.publish_raw_car_control([v_left, v_right, v_left, v_right])
            time.sleep(dt)

        self.stop()
        pose = self.get_current_pose()
        if pose:
            self.node.get_logger().warn(
                f"⚠️ 旋轉超時！目標: {math.degrees(target_yaw):.1f}°, "
                f"實際: {math.degrees(pose[2]):.1f}°"
            )
        return False

    # ==================================================================
    #  PID 直線行走
    # ==================================================================

    def drive_straight(self, distance, timeout=180.0, stop_at_wall_dist=None, min_travel_for_wall=0.0):
        """
        使用 PID 控制器直線前進指定距離 (公尺)。
        使用「目標點絕對座標」做到達判定，不受斜坡打滑影響。
        同時用航向 PID 修正偏航，保持直線。

        Args:
            distance: 前進距離 (m)，正值前進，負值後退
            timeout: 超時秒數
            stop_at_wall_dist: 若設定，當雷射測距前方小於此值時提早停止 (防撞牆)
            min_travel_for_wall: 至少要走多遠才啟用防撞牆 (避免在上坡時雷射掃到斜坡誤觸)
        Returns:
            True if reached, False if timeout
        """
        # 等待取得起始位姿
        start_pose = None
        t0 = time.time()
        while start_pose is None and time.time() - t0 < 5.0:
            start_pose = self.get_current_pose()
            time.sleep(0.05)
        if start_pose is None:
            self.node.get_logger().error("drive_straight: 無法取得起始位姿")
            return False

        start_x, start_y, start_yaw = start_pose
        target_yaw = start_yaw  # 維持啟動時的朝向

        # 計算目標座標（絕對位置）
        target_x = start_x + distance * math.cos(start_yaw)
        target_y = start_y + distance * math.sin(start_yaw)

        self.node.get_logger().info(
            f"🚗 開始直線行走: 距離 {distance:.2f}m, "
            f"起點 ({start_x:.2f}, {start_y:.2f}), "
            f"目標 ({target_x:.2f}, {target_y:.2f}), "
            f"航向 {math.degrees(start_yaw):.1f}°"
        )

        # PID 參數
        Kp_dist = 400.0
        Ki_dist = 8.0
        Kd_dist = 30.0

        Kp_yaw = 300.0
        Kd_yaw = 30.0

        MIN_DRIVE_SPEED = 220.0
        MAX_DRIVE_SPEED = 400.0
        ARRIVE_TOLERANCE = 0.20  # 到達判定距離 (m)

        integral_dist = 0.0
        prev_error_dist = None
        prev_yaw_error = 0.0
        start_time = time.time()
        dt = 0.05

        on_target_count = 0
        ON_TARGET_REQUIRED = 4

        direction = 1.0 if distance >= 0 else -1.0
        last_log_time = 0
        drive_time_since_correction = 0.0
        max_traveled = 0.0

        while rclpy.ok() and (time.time() - start_time) < timeout:
            pose = self.get_current_pose()
            if pose is None:
                time.sleep(dt)
                continue

            cx, cy, cyaw = pose

            # ⭐ 用到目標點的絕對距離做判定（不受打滑影響）
            dist_to_target = math.sqrt((target_x - cx)**2 + (target_y - cy)**2)

            dx = cx - start_x
            dy = cy - start_y
            traveled = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)

            # 處理 AMCL 異常跳退 (例如下橋時將地板誤認為牆壁)
            if traveled < max_traveled - 0.1:
                # 若 AMCL 退後超過 0.1m，我們不採信倒退的座標，直接用歷史最遠紀錄
                traveled = max_traveled
            else:
                if traveled > max_traveled:
                    max_traveled = traveled

            remaining = distance - traveled

            # 防撞牆機制 & 雷射圖資同步機制
            if stop_at_wall_dist is not None and max_traveled >= min_travel_for_wall:
                lidar = self.node.get_latest_lidar()
                if lidar is not None:
                    ranges = lidar.ranges
                    # 0 度在 index 0，所以正前方是陣列的最前面與最後面
                    front_ranges = ranges[-10:] + ranges[:10]
                    # 放寬下限到 0.01，避免已經貼牆時測距太近被過濾掉而失效
                    valid = [r for r in front_ranges if 0.01 < r < 10.0]
                    if valid:
                        front_dist = min(valid)
                        
                        # 核心同步邏輯：用光達距離反推真正的剩餘距離！
                        # 目標是停在離牆 stop_at_wall_dist 的地方，所以真實剩餘距離 = 光達距離 - 目標離牆距離
                        true_remaining = front_dist - stop_at_wall_dist
                        sync_traveled = distance - true_remaining
                        
                        # 如果光達推算出的進度比 AMCL 算出來的還要快（代表 AMCL 卡住了），我們強制同步！
                        if sync_traveled > traveled:
                            traveled = sync_traveled
                            if traveled > max_traveled:
                                max_traveled = traveled
                            remaining = true_remaining
                            dist_to_target = abs(true_remaining)

                            # --- 執行真正的 ROS 地圖同步 (AMCL Sync) ---
                            # 計算此進度下真實的絕對座標
                            true_x = start_x + sync_traveled * math.cos(start_yaw)
                            true_y = start_y + sync_traveled * math.sin(start_yaw)
                            
                            sync_msg = PoseWithCovarianceStamped()
                            sync_msg.header.frame_id = 'map'
                            sync_msg.header.stamp = self.node.get_clock().now().to_msg()
                            sync_msg.pose.pose.position.x = true_x
                            sync_msg.pose.pose.position.y = true_y
                            # 把 start_yaw 轉成四元數 (我們只考慮 2D 旋轉)
                            sync_msg.pose.pose.orientation.z = math.sin(start_yaw * 0.5)
                            sync_msg.pose.pose.orientation.w = math.cos(start_yaw * 0.5)
                            # 設定一個極小的變異數讓 AMCL 強制信服這個座標
                            sync_msg.pose.covariance[0] = 0.01
                            sync_msg.pose.covariance[7] = 0.01
                            sync_msg.pose.covariance[35] = 0.01
                            
                            self.initial_pose_pub.publish(sync_msg)

                        if front_dist <= stop_at_wall_dist:
                            self.stop()
                            self.node.get_logger().info(
                                f"🛑 觸發雷射同步防撞！前方真實距離 {front_dist:.2f}m (設定 <= {stop_at_wall_dist}m)，"
                                f"強制完成！"
                            )
                            return True

            # 每行駛約 0.5 公尺 (用時間推算約 2.5 秒)，暫停一下讓 AMCL 重新收斂校正
            drive_time_since_correction += dt
            if drive_time_since_correction >= 2.5:
                self.stop()
                self.node.get_logger().info(f"🔄 已行駛一段時間 (目前已走 {traveled:.2f}m)，暫停 2 秒讓光達掃描與 AMCL 校正...")
                time.sleep(2.0)
                drive_time_since_correction = 0.0
                
                # 重新讀取校正後的 pose
                new_pose = self.get_current_pose()
                if new_pose:
                    cx, cy, cyaw = new_pose
                    dist_to_target = math.sqrt((target_x - cx)**2 + (target_y - cy)**2)
                    self.node.get_logger().info(f"✅ 校正完成！目前位置 ({cx:.2f}, {cy:.2f})，離目標還有 {dist_to_target:.2f}m")
                # 不跳過迴圈，繼續往下計算 PID
                
            # 每 3 秒報告一次進度
            now = time.time()
            if now - last_log_time > 3.0:
                self.node.get_logger().info(
                    f"  📍 位置 ({cx:.2f}, {cy:.2f}), "
                    f"離目標 {dist_to_target:.2f}m, 已走 {traveled:.2f}m"
                )
                last_log_time = now

            # 到達判定：用到目標點的絕對距離
            if dist_to_target < ARRIVE_TOLERANCE:
                on_target_count += 1
                if on_target_count >= ON_TARGET_REQUIRED:
                    self.stop()
                    self.node.get_logger().info(
                        f"✅ 直線行走完成！離目標 {dist_to_target:.2f}m, "
                        f"到達 ({cx:.2f}, {cy:.2f})"
                    )
                    return True
            else:
                on_target_count = 0

            # 距離 PID（用剩餘距離驅動，保證在斜坡上也能持續前進）
            error_dist = dist_to_target * direction
            # 如果已經超過目標（traveled > distance），error 應該變負讓車子減速
            if direction > 0 and remaining < 0:
                error_dist = remaining
            elif direction < 0 and remaining > 0:
                error_dist = remaining

            integral_dist += error_dist * dt
            # 放寬距離積分上限，以應付上橋時需要的額外持續推力
            integral_dist = max(-10.0, min(10.0, integral_dist))

            if prev_error_dist is not None:
                derivative_dist = (error_dist - prev_error_dist) / dt
            else:
                derivative_dist = 0.0
            prev_error_dist = error_dist

            speed = Kp_dist * error_dist + Ki_dist * integral_dist + Kd_dist * derivative_dist

            # 限制速度範圍
            if abs(speed) > MAX_DRIVE_SPEED:
                speed = MAX_DRIVE_SPEED if speed > 0 else -MAX_DRIVE_SPEED
            elif abs(speed) < MIN_DRIVE_SPEED and dist_to_target > ARRIVE_TOLERANCE:
                speed = MIN_DRIVE_SPEED * direction

            # 航向修正 PID（保持直線）
            yaw_error = self.normalize_angle(target_yaw - cyaw)
            yaw_derivative = (yaw_error - prev_yaw_error) / dt
            prev_yaw_error = yaw_error
            yaw_correction = Kp_yaw * yaw_error + Kd_yaw * yaw_derivative
            yaw_correction = max(-150.0, min(150.0, yaw_correction))

            # 合成左右輪速度
            v_left = speed - yaw_correction
            v_right = speed + yaw_correction

            # 最終安全限幅
            max_val = max(abs(v_left), abs(v_right))
            if max_val > MAX_DRIVE_SPEED:
                scale = MAX_DRIVE_SPEED / max_val
                v_left *= scale
                v_right *= scale

            self.node.publish_raw_car_control([v_left, v_right, v_left, v_right])
            time.sleep(dt)

        self.stop()
        pose = self.get_current_pose()
        if pose:
            dist_to_target = math.sqrt((target_x - pose[0])**2 + (target_y - pose[1])**2)
            self.node.get_logger().warn(
                f"⚠️ 直線行走超時！離目標還有 {dist_to_target:.2f}m, 位置 ({pose[0]:.2f}, {pose[1]:.2f})"
            )
        return False

    def execute_yolo_door_sequence(self):
        self.node.get_logger().info("🔍 啟動 YOLO 對齊與開門程序！")
        # 等待 YOLO 有輸出
        time.sleep(2.0)

        # =============================================================
        # 階段 0: 掃描尋找門把 (利用 rotate_to_yaw 左右擺動)
        # =============================================================
        self.node.get_logger().info("👀 開始掃描尋找門把 (左右擺動)...")
        SCAN_ANGLE_DEG = 20.0    # 每次左右掃描的角度
        SCAN_ROUNDS = 3          # 來回掃幾輪
        found = False

        # 取得目前朝向作為基準
        pose = self.get_current_pose()
        if pose is None:
            self.node.get_logger().warn("⚠️ 無法取得當前位姿")
            self.unlock_door()
            self.clear_door()
            return
        base_yaw = pose[2]

        # 掃描模式: 先檢查目前方向，再左轉、右轉交替
        scan_offsets_deg = [0]  # 先看正前方
        for i in range(1, SCAN_ROUNDS + 1):
            scan_offsets_deg.append(SCAN_ANGLE_DEG * i)   # 左
            scan_offsets_deg.append(-SCAN_ANGLE_DEG * i)  # 右

        for offset_deg in scan_offsets_deg:
            # 檢查 YOLO 有沒有已經看到
            info = self.node.get_latest_yolo_target_info()
            if info is not None and len(info.data) >= 3 and info.data[0] >= 0.5:
                self.stop()
                found = True
                self.node.get_logger().info(f"🎯 在偏移 {offset_deg:.0f}° 處發現門把！")
                break

            # 旋轉到目標角度
            target_yaw = self.normalize_angle(base_yaw + math.radians(offset_deg))
            self.node.get_logger().info(f"🔄 掃描: 旋轉到 {math.degrees(target_yaw):.1f}° (偏移 {offset_deg:.0f}°)")
            self.rotate_to_yaw(target_yaw, timeout=8.0)
            time.sleep(0.5)

            # 旋轉完再次檢查
            info = self.node.get_latest_yolo_target_info()
            if info is not None and len(info.data) >= 3 and info.data[0] >= 0.5:
                self.stop()
                found = True
                self.node.get_logger().info(f"🎯 在偏移 {offset_deg:.0f}° 處發現門把！")
                break

        if not found:
            self.stop()
            self.node.get_logger().warn("⚠️ 掃描超時仍未找到門把，嘗試直接開門")
            self.unlock_door()
            self.clear_door()
            return

        # =============================================================
        # 階段 1: 精準旋轉對齊門把 (用 rotate_to_yaw 逐步微調)
        # =============================================================
        self.node.get_logger().info("🔄 開始 YOLO 精準對齊 (Rotation)...")
        aligned = False
        # 像素偏移 → 角度的粗略換算係數
        # 假設相機水平 FOV ≈ 60°, 解析度寬 ≈ 640px → 每 pixel ≈ 0.094°
        PIXEL_TO_DEG = 0.094
        MAX_ATTEMPTS = 8

        for attempt in range(MAX_ATTEMPTS):
            time.sleep(0.5)
            info = self.node.get_latest_yolo_target_info()
            if info is None or len(info.data) < 3 or info.data[0] < 0.5:
                self.node.get_logger().info(f"🔍 對齊嘗試 {attempt+1}: 目標丟失，等待...")
                time.sleep(1.0)
                continue

            delta_x = info.data[2]
            self.node.get_logger().info(
                f"🎯 對齊嘗試 {attempt+1}: delta_x = {delta_x:.1f} px"
            )

            if abs(delta_x) < 30.0:
                aligned = True
                self.node.get_logger().info("✅ 門把對齊完成！")
                break

            # 用 delta_x 估算需要旋轉的角度 (正 delta_x = 門把在右邊 → 需右轉 = yaw 減少)
            angle_correction_deg = -delta_x * PIXEL_TO_DEG
            # 限制單次修正幅度，避免過衝
            angle_correction_deg = max(-15.0, min(15.0, angle_correction_deg))

            pose = self.get_current_pose()
            if pose is None:
                continue
            target_yaw = self.normalize_angle(pose[2] + math.radians(angle_correction_deg))
            self.node.get_logger().info(
                f"  → 修正 {angle_correction_deg:.1f}°, 旋轉到 {math.degrees(target_yaw):.1f}°"
            )
            self.rotate_to_yaw(target_yaw, timeout=5.0)

        if not aligned:
            self.stop()
            self.node.get_logger().warn("⚠️ 無法精準對齊門把，直接嘗試開門")

        # =============================================================
        # 階段 2: 讀取 YOLO 深度，計算往前距離
        # =============================================================
        time.sleep(0.5)
        info = self.node.get_latest_yolo_target_info()
        if info is not None and len(info.data) >= 2:
            target_depth = info.data[1]
            self.node.get_logger().info(f"📏 讀取到門把深度: {target_depth:.2f}m")
        else:
            target_depth = 0.35
            self.node.get_logger().warn(f"⚠️ 無法讀取深度，使用預設值 {target_depth:.2f}m")

        # =============================================================
        # 階段 3: 舉起手臂 (保持不放下)
        # =============================================================
        self.node.get_logger().info("🦾 舉起手臂到開門位置 (保持)...")
        # x_offset 往前伸，z_offset 往上抬高確保能壓到門把
        self.arm_controller.move_end_effector(x_offset=0.20, y_offset=0.0, z_offset=0.10)
        time.sleep(0.5)
        self.node.get_logger().info("✅ 手臂已就位，保持伸出狀態")

        # =============================================================
        # 階段 4: 帶著伸出的手臂往前行駛
        # =============================================================
        drive_distance = max(0.0, target_depth - 0.20)
        self.node.get_logger().info(f"🚗 手臂保持伸出，往前行駛 {drive_distance:.2f}m (深度減去 0.20m)...")
        # 直接使用 PID 直線行駛，確保走到指定深度
        if drive_distance > 0:
            self.drive_straight(drive_distance)
        time.sleep(0.5)

        # =============================================================
        # 階段 5: 大幅度上下揮動手臂壓門把
        # =============================================================
        self.node.get_logger().info("🔓 抵達門把，開始上下揮動手臂嘗試開門...")
        for i in range(3):
            self.node.get_logger().info(f"  > 揮動次數 {i+1}/3: 往下壓")
            self.arm_controller.move_end_effector(x_offset=0.20, y_offset=0.0, z_offset=-0.15)
            time.sleep(0.8)
            
            self.node.get_logger().info(f"  > 揮動次數 {i+1}/3: 往上抬")
            self.arm_controller.move_end_effector(x_offset=0.20, y_offset=0.0, z_offset=0.15)
            time.sleep(0.8)

        self.arm_controller.ensure_joint_pos_initialized()
        time.sleep(1.0)
        
        # =============================================================
        # 階段 6: 精準導航通過門
        # =============================================================
        self.node.get_logger().info("🚀 使用 PID 直線行駛通過門！(前進 2.0m)")
        self.drive_straight(2.0)

    # ==================================================================
    #  門相關動作
    # ==================================================================

    def unlock_door(self):
        self.node.get_logger().info("執行推開門把動作！")
        self.arm_controller.move_end_effector(x_offset=0.15, y_offset=0.0, z_offset=0.0)
        time.sleep(2)
        self.arm_controller.ensure_joint_pos_initialized()

    def clear_door(self):
        self.node.get_logger().info("直線衝刺通過門！")
        VEL = 600.0
        for _ in range(30):
            self.node.publish_raw_car_control([VEL, VEL, VEL, VEL])
            time.sleep(0.1)
        self.node.publish_raw_car_control([0.0, 0.0, 0.0, 0.0])


def to_global(lx, ly, lyaw, x0, y0, yaw0):
    """
    將相對於起始點的「局部座標 (lx, ly)」與「相對朝向 lyaw」
    映射到世界座標上的絕對位置與絕對朝向。
    """
    gx = x0 + lx * math.cos(yaw0) - ly * math.sin(yaw0)
    gy = y0 + lx * math.sin(yaw0) + ly * math.cos(yaw0)
    gyaw = yaw0 + lyaw
    while gyaw > math.pi: gyaw -= 2 * math.pi
    while gyaw < -math.pi: gyaw += 2 * math.pi
    return gx, gy, gyaw


def main(args=None):
    rclpy.init(args=args)
    runner = FinalTaskRunner()

    runner.node.get_logger().info("啟動 Final Task Runner (方案B：完全手動 PID 控制)！")
    time.sleep(2)

    # 確保 AMCL 有初始值
    runner.publish_initial_pose()
    time.sleep(1)

    runner.arm_controller.ensure_joint_pos_initialized()
    time.sleep(2)

    # =============================================================
    # 1. 取得起始絕對座標
    # =============================================================
    start_pose = None
    t0 = time.time()
    while start_pose is None and time.time() - t0 < 10.0:
        start_pose = runner.get_current_pose()
        time.sleep(0.1)

    if start_pose is None:
        runner.node.get_logger().error("無法取得起始 AMCL 位姿，程式終止。")
        rclpy.shutdown()
        return

    x0, y0, yaw0 = start_pose
    runner.node.get_logger().info(f"✅ 獲取到基準絕對座標: X={x0:.2f}, Y={y0:.2f}, Yaw={math.degrees(yaw0):.1f}°")

    # =============================================================
    # 2. 定義任務序列 (完全手動 PID 控制)
    #
    #    動作路徑推演 (相對於起點)：
    #    起點 (0,0) 面朝 +x
    #    1.  直走 1.0m → 到 (1.0, 0.0)，面朝 +x
    #    2.  停留 5 秒
    #    3.  原地左轉 90° → 面朝 +y
    #    4.  直走過橋 3.0m → 到 (1.0, 3.0)，面朝 +y
    #    5.  原地右轉 90° → 面朝 +x
    #    6.  直走 0.7m → 到 (1.7, 3.0)，面朝 +x
    #    7.  原地右轉 80° → 面朝 -80°
    #    8.  直走 1.5m → 到 (1.96, 1.52)，面朝 -80°
    #    9.  原地左轉 80° → 面朝 +x (0°)
    #    10. 直走 1.5m → 到 (3.46, 1.52)，面朝 +x
    #    11. 開門
    # =============================================================

    sequence = [
        {'action': 'drive',  'desc': '1. 往前 1.0m',       'distance': 1.0},
        {'action': 'sleep',  'desc': '2. 停留 5 秒',       'time': 5.0},
        {'action': 'turn',   'desc': '3. 原地左轉 90°',   'yaw': math.pi/2},
        # 防撞牆必須走過 1.0m (越過上坡) 才啟用，避免把上坡當作牆壁
        {'action': 'drive',  'desc': '4. 直走過橋 3.0m',       'distance': 3.0, 'stop_at_wall': 0.8, 'min_travel_for_wall': 1.0},
        {'action': 'turn',   'desc': '5. 原地右轉 90°',   'yaw': 0.0},
        {'action': 'drive',  'desc': '6. 直走 0.7m',       'distance': 0.7},
        {'action': 'turn',   'desc': '7. 原地右轉 80°',   'yaw': math.radians(-80)},
        {'action': 'drive',  'desc': '8. 直走 1.5m',       'distance': 1.5},
        {'action': 'turn',   'desc': '9. 原地左轉 80°',   'yaw': 0.0},
        {'action': 'drive',  'desc': '10. 直走 1.5m',      'distance': 1.5},
        {'action': 'yolo_door','desc': '11. YOLO 對齊並開門'}
    ]

    # =============================================================
    # 3. 執行任務序列
    # =============================================================
    runner.start_recording()

    # 追蹤累計的相對朝向，用於 rotate 動作的全局目標計算
    current_relative_yaw = 0.0

    try:
        runner.node.get_logger().info("===== 開始執行任務序列 (完全手動 PID) =====")
        for step in sequence:
            runner.node.get_logger().info(f"執行步驟: {step['desc']}")

            if step['action'] == 'turn':
                # 將相對朝向轉換為全局朝向
                target_relative_yaw = step['yaw']
                _, _, gyaw = to_global(0, 0, target_relative_yaw, x0, y0, yaw0)
                runner.node.get_logger().info(
                    f"  > 目標全局朝向: {math.degrees(gyaw):.1f}°"
                )
                success = runner.rotate_to_yaw(gyaw)
                if not success:
                    runner.node.get_logger().warn("⚠️ 旋轉未達標，但程式繼續。")
                time.sleep(1.0)

            elif step['action'] == 'drive':
                dist = step['distance']
                stop_at_wall = step.get('stop_at_wall', None)
                min_travel = step.get('min_travel_for_wall', 0.0)
                runner.drive_straight(dist, stop_at_wall_dist=stop_at_wall, min_travel_for_wall=min_travel)
                time.sleep(1.0)

            elif step['action'] == 'sleep':
                time.sleep(step['time'])

            elif step['action'] == 'yolo_door':
                runner.execute_yolo_door_sequence()

        runner.node.get_logger().info("===== 所有任務完成！ =====")
        runner.stop_recording_and_save("/workspaces/launch/robot_trajectory.json")
        time.sleep(1) # 短暫等待讓 ROS log 輸出完畢

    except KeyboardInterrupt:
        pass
    finally:
        if runner.is_recording:
            runner.stop_recording_and_save("/workspaces/launch/robot_trajectory.json")
            
        try:
            runner.stop()
        except Exception as e:
            pass
            
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()