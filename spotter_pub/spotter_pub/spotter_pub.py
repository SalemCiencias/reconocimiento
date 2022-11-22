import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import models, transforms
from std_msgs.msg import String
import torch.nn as nn
import json
import sys


class Spotter(Node):

    """
    Publicador que se encarga de clasificar imagenes recibidas porla camara de 
    vision
    """

    def __init__(self):
        """
        Se crea un nodo llamado spotter que se suscribe al topico video frames
        y clasifica cada imagen recibida de acuerdo al modelo proporcionado.
        Publica las etiquetas que logró reconocer el nodo
        """
        super().__init__('spotter')
        self.subscription = self.create_subscription(Image, 'video_frames',
                                                     self.listener_callback,
                                                     10)
        self.publisher_ = self.create_publisher(String, 'infront', 10)
        self.br = CvBridge()
        self.subscription
        self.recon_model = recognition_model()
        self.recon_model.new_model()

    def listener_callback(self, data):

        self.get_logger().info('Recibiendo imagen')

        current_frame = self.br.imgmsg_to_cv2(data)
        tag = self.recon_model.get_class(current_frame)

        cv2.putText(current_frame, tag, (160, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        msg = String()
        msg.data = tag
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)

        cv2.imshow("camera", current_frame)
        cv2.waitKey(1)


class recognition_model():
    ACTIVE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def __init__(self, tolerance=.50) -> None:
        self.model = None
        self.DEVICE = recognition_model.ACTIVE_DEVICE
        self.tags = None
        self.tolerance = tolerance

    def initialize_model(num_classes: int):
        '''
        Ajusta el modelo pre-entrenado a nuestras necesidades actuales
        '''
        model_ft = None
        input_size = 0

        model_ft = models.squeezenet1_0(
            weights=models.SqueezeNet1_0_Weights.DEFAULT)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

        return model_ft, input_size

    def preprocess():
        """
        Prepara una imagen para ajustarse a la entrada de la red 224 x 224 x 3
        (imagenes de 224 x 224 de 3 canales de color RGB)
        Y normalizarla a los valores pedidos en la documentación
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), ])
        return transform

    def new_model(self, TAGS='resource/tags/tags.json', MODEL="resource/models/model.pt"):
        '''
        Carga un nuevo  modelo de reconocimiento
        '''
        self.tags: dict = {}

        with open(TAGS, 'r') as openfile:
            self.tags = json.load(openfile)

        if self.tags is not {}:
            num_classes = len(self.tags.keys())
        else:
            print('No se encontraron las etiquetas, abortando...')
            sys.exit(-1)

        try:
            self.model, input_size = recognition_model.initialize_model(
                num_classes)

            self.model.load_state_dict(torch.load(
                MODEL, map_location=torch.device(recognition_model.ACTIVE_DEVICE)))
        except Exception as e:
            print(e)
            print('No se pudo cargar el modelo con exito, abortando')
            sys.exit(-1)

        self.model.eval()

    def get_class(self, image):
        '''
        retorna una cadena que etiqueta la imagen recibida por parámetro
        '''
        transform = recognition_model.preprocess()
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top_prob, top_catid = torch.topk(probabilities, 1)
        print(top_catid)
        top_catid = top_catid.numpy()
        if top_prob > self.tolerance:
            return self.tags[str(top_catid[0])]

        return "No indentify"


def main(args=None):

    rclpy.init(args=args)

    spotter = Spotter()

    rclpy.spin(spotter)
    spotter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()