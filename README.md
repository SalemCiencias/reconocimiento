# Reconocimiento
Modulos que auxilian en el reconocimiento de imagenes

## webcam_sub

Este modulo provee un publicador que se conecta a la web cam y publica imagenes tomadas por esta. Tambien incluye un ejemplo de subscritor al tópico `video_frames` 

- webcam_pub.py : publicador de las imagenes tomadas por la web cam
- webcam_sub.py : ejemplo de subscritor al topico `video_frames` 

## spotter_pub

Este módulo se subscribe al topico `video_frames` y se encarga de clasificar las imagenes en cuatro categorias: sillas, puertas, personas y no identify. Y publica la clasificación bajo el tópico `infront`

**warning** : Spotter tiene dependencias de ambiente no satisfacibles por rosdep, pero se incluye un archivo [requeriments.txt](https://github.com/SalemCiencias/reconocimiento/blob/main/spotter_pub/requeriments.txt) que se puee usar para instalarlas usando:

```
pip install -r requeriments.txt
```

## Rentrenamiento para spotter

Si spotter no está realizando una clasificación de manera correcta, puede rentrenar un nuevo modelo usando los paquetes en [ShowMeWhatUSee](https://github.com/cocisran/Show-me-what-you-see) y cambiando el modelo de clasificacion en la carpeta resourcers.
