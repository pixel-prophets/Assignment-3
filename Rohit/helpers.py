from firebase_admin import credentials, initialize_app, storage, db
import uuid

cred = credentials.Certificate("./dv-assg-3-creds.json")
initialize_app(cred, {
    'storageBucket': 'dv-assg-3.appspot.com', 
    'databaseURL': 'https://dv-assg-3-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

def uploadImageToFirebase(path: str, name: str, step: str) -> str:
    unique_id = uuid.uuid4()
    fileName = f"{path}_{uuid}" 
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(path)

    # ! Opt : if you want to make public access from the URL
    blob.make_public()

    ref = db.reference('/images')

    ref.push({
        'name': name,
        'url': blob.public_url,
        'step': step
    })

    return blob.public_url

uploadImageToFirebase("tmp.png", "1.jpg", "1")