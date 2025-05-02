import matplotlib.pyplot as plt

models = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5']
params = [4.3, 6.8, 8.0, 11.0, 17.9, 28.8]
accuracy = [42.30, 39.96, 40.22, 40.68, 38.46, 41.49]

plt.figure(figsize=(8, 6))
plt.plot(params, accuracy, 'o-', label='EfficientNet models', markersize=8)

for i, txt in enumerate(models):
    plt.annotate(txt, (params[i], accuracy[i]), textcoords="offset points", xytext=(5,5), ha='left')

plt.xlabel('Number of Parameters (Millions)')
plt.ylabel('Validation Accuracy (%)')
plt.title('Size vs. Accuracy Trade-off (Tiny-ImageNet-200)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.savefig("Size vs. Accuracy Trade-off (Tiny-ImageNet-200).svg", format='svg')




# import matplotlib.pyplot as plt

# # 모델별 FLOPS (GFLOPs)와 Accuracy (%)
# models = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5']
# flops = [0.39, 0.7, 1.0, 1.8, 4.2, 9.9, 37.0]  # FLOPS in GFLOPs
# accuracy = [77.1, 79.1, 80.3, 81.7, 83.0, 83.6, 84.3]  # Top-1 Accuracy

# plt.figure(figsize=(8, 6))
# plt.plot(flops, accuracy, marker='o', linestyle='-', color='red')

# for i, model in enumerate(models):
#     plt.text(flops[i], accuracy[i] + 0.3, model, fontsize=9)

# plt.xlabel('FLOPS (GFLOPs)')
# plt.ylabel('Top-1 Accuracy (%)')
# plt.title('EfficientNet FLOPS vs Accuracy')
# plt.grid(True)
# plt.show()
