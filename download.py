import timm

print('downloading')
model = timm.create_model(
    'vit_large_patch16_dinov3.lvd1689m',
    pretrained=True,
    features_only=True,
)
model = model.eval()
print('model downloaded')

