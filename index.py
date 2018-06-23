import turicreate as tc
import os

def generate_sframe():
    data = tc.image_analysis.load_images('DatasetTraining', with_path=True)
    data['number'] = data['path'].apply(lambda path: os.path.basename(os.path.dirname(path)))
    data.save('turimodel.sframe')
    data.explore

def train_sframe_and_generate_mlmodel():
    data = tc.SFrame('turimodel.sframe')
    train_data, test_data = data.random_split(0.85)
    model = tc.image_classifier.create(train_data, target='number')
    predictions = model.predict(test_data)
    stats = model.evaluate(test_data)
    print('metrics accuracy is ' + stats['accuracy'])
    model.save('turi.model')
    model.export_coreml('model/MyModel.mlmodel')

generate_sframe()
train_sframe()