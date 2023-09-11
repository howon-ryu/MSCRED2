import numpy as np
import matplotlib.pyplot as plt
import os
import util


file_path = r'D:\MSCRED2\data\synthetic_data_with_anomaly-s-1.csv'
all_data = np.loadtxt(file_path, delimiter=',')
all_data = all_data.T
print(all_data.shape)



# print(all_data[11790:11815,15])


# plt.plot(all_data[11650:11950,15], color='blue', linestyle='-', marker='o')
# plt.xlim(0, 300)
# plt.axvspan(150, 160, color='red', linewidth=2)



def create_anomaly(data):
        root_cause_f = open(r"D:\MSCRED2\data\test_anomaly.csv", "r")

        root_cause_gt = np.loadtxt(root_cause_f, delimiter=",").astype(np.int64)
        anomaly_pos = root_cause_gt[:, 0]
        print("anomaly_pos",anomaly_pos)
        for i in range(5):
            
            anomaly_series = [root_cause_gt[:,i] for i in range(1,4)]
            print("anomaly_series",anomaly_series)
            for k in range(3):
                for j in anomaly_series[k]:
                    
                    
                    
                    base_value = data[anomaly_pos[i], j]
                    print("base_value",anomaly_pos[i],j)
                    #print("prev",data[anomaly_pos[i]-10:anomaly_pos[i] + 10, 15])
                    data[anomaly_pos[i]-10:anomaly_pos[i] , j] = base_value + np.random.normal(loc=5, scale=0.8, size=10)
                    #print("post",data[11800:11820, 15])
        return data

add_ano = create_anomaly(all_data)
print("add_ano",add_ano)
print(add_ano[11800:11820,15])
plt.plot(add_ano[11650:11950,15], color='blue', linestyle='-', marker='o')
plt.plot(add_ano[11650:11950,24], color='green', linestyle='-', marker='o')
plt.xlim(0, 300)
plt.axvspan(150, 160, color='red', linewidth=2)


plt.show()

# 데이터 로드 및 형태 확인
test_data_path = util.test_data_path
reconstructed_data_path = util.reconstructed_data_path
test_data_path = os.path.join(test_data_path, "test.npy")
reconstructed_data_path = os.path.join(reconstructed_data_path, "test_reconstructed.npy")
test_data = np.load(test_data_path)
test_data = test_data[:, -1, ...]  # only compare the last matrix with the reconstructed data
reconstructed_data = np.load(reconstructed_data_path)
print("test_data",test_data.shape)











# # plot anomaly score curve and identification result
# anomaly_pos = np.zeros(5)
# root_cause_gt = np.zeros((5, 3))
# anomaly_span = [10, 30, 90]

# # Read the test_anomaly.csv, each line behalf of an anomaly, the first is the position, the next three number is the
# # root cause.
# root_cause_f = open(r"D:\MSCRED2\data\test_anomaly.csv", "r")

# root_cause_gt = np.loadtxt(root_cause_f, delimiter=",").astype(np.int64)
# anomaly_pos = root_cause_gt[:, 1]

# print("anomaly_pos",anomaly_pos)

# anomaly_pos = [(anomaly_pos[i]/util.gap_time-util.test_start_id-anomaly_span[i % 3]/util.gap_time) for i in range(5)]

# # 데이터를 1D 배열로 축소 (평균 계산)
# test_data_1d = np.mean(test_data, axis=(1, 2, 3))
# restructed_data_1d = np.mean(reconstructed_data, axis=(1, 2, 3))

# # 데이터 시각화
# plt.figure(figsize=(16, 8))
# plt.plot(test_data_1d, label='Test Data')
# plt.plot(restructed_data_1d, label='Reconstructed Data')
# for k in range(len(anomaly_pos)):
#     plt.axvspan(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k % 3] / util.gap_time, color='red', linewidth=2)

# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')
# plt.title('Average Synthetic Data')
# plt.legend()  # 범례 추가
# plt.show()