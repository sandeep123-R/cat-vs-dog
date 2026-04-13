import tensorflow as tf

# 1. Rebuild architecture
base_model = tf.keras.applications.MobileNet(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# 2. LOAD WEIGHTS ONLY (VERY IMPORTANT)
model = tf.keras.models.load_model("final_clean_model.h5")

# 3. SAVE CLEAN MODEL
model.save("final_clean_model.h5", include_optimizer=False)

print("✅ CLEAN MODEL CREATED")