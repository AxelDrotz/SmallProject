
# X = np.load('normalized_array.npy').transpose()
# # k_dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50,
# #                60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 750]
# k_dimensions = [1, 2]
# rp_error_results, srp_error_results, rp_time, srp_time = srp_test(
#     X, k_dimensions)

# fig, axis = plt.subplots(2, 2)
# axis[0, 0].plot(k_dimensions, rp_error_results)
# axis[0, 0].set_title("Random Projection")
# axis[0, 1].plot(k_dimensions, srp_error_results)
# axis[0, 1].set_title("Sparse Random Projection")
# axis[1, 0].set_yscale("log")
# axis[1, 0].plot(k_dimensions, rp_time)
# axis[1, 0].set_title("Random Projection Elapsed Time")
# axis[1, 1].set_yscale("log")
# axis[1, 1].plot(k_dimensions, srp_time)
# axis[1, 1].set_title("Sparse Random Projection Elapsed Time")
# plt.show()
