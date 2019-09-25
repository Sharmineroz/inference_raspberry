# Clasificador en Raspberry Pi

###### Este es un ejemplo de implementaci�n de clasificador usando CNN como clasificador en la tarjeta Raspberry Pi 3 model b+.

### Conexi�n SSH (opcional)

Para comenzar, vamos a establecer comunicaci�n SSH entre nuestro computador y la tarjeta, este paso es opccional, pero deja m�s recursos disponibles en la tarjeta para usarlos en procesamiento.

Para ello, abrimos el terminal y creamos un archivo llamado "ssh" en el directorio /boot:

`sudo touch ../../boot/ssh`

Posteriormente, entramos a la configurafi�n interfaz de las raspberry para activar la conexi�n SSH y de paso la c�mara (que se usar� despu�s):

`sudo raspi-config`

![](https://www.raspberrypi-spy.co.uk/wp-content/uploads/2012/05/pi_configuration_interfacing_options.png)

Entre a opciones de interfaz y habilite las opciones de SSH y c�mara.

![](https://www.raspberrypi-spy.co.uk/wp-content/uploads/2012/05/pi_configuration_interfacing_options_ssh.png)

Luego selecci�ne "OK" seguido de  "YES" y cuando haya activado las dos interfaces, seleccione "Finish".

Para conectarse a la raspberry desde su pc, use el programa [putty](https://the.earth.li/~sgtatham/putty/latest/w64/putty.exe "putty"), para ello, entre en la carpeta de descargas de su computador y abra un terminal de comandos, escriba ah� `putty.exe` seguido de la ip a la que est� conectada la raspberry (para saberla use en a raspberry `ifconfig`), ejemplo:

`putty.exe 192.168.0.2`

En la ventana emergente, entre con el usuario `pi` y la clave que le haya puesto a su tarjeta (por defecto es `raspberry`). Con esto ya est� establecida la conexi�n SSH. 

### Configuraci�n de entorno virtual (opcional)

Ahora nos cambiamos al escritorio, creamos una carpeta en la que vamos a alojar el el proyecto y nos situamos en esa carpeta: 

`cd Desktop`
`mkdir inferencia`
`cd inferencia`

Actualizamos a la versi�n m�s reciente de python:

`sudo apt-get install python2.7-dev python3-dev`

Ahora vamos a descargar virtualenv para crear un entorno virtual en el que vamos a guardar las librer�as necesarias para el proyecto. 

`sudo apt install virtualenv python3-virtualenv -y`
`virtualenv -p /usr/bin/python3 tf`
`source tf/bin/activate`

### Configuraci�n de picamera 

Vamos a probar la c�mara, con�ctela como se muestra en este [video](http://https://youtu.be/GImeVqHQzsE "video") y en el terminal escriba:

`raspistill -o output.jpg`

Compruebe si se ha tomado la im�gen mediante `ls`, deber�a ver `output.jpg` en la respuesta.

Posteriormente, instalamos la librer�a de manejo de los datos de la c�mara para python. Asegurese de que es `picamera[array]` y no `picamera`.

`pip3 install "picamera[array]"`

Compruebe que ha quedado bien instalado con:

`python`
`>>> import PiRGBArray`

### Instalaci�n de Tensorflow

Ahora vamos a instalar tensorflow, para ello, primero instalamos un paquete necesario para que funcione en nuestra tarjeta y luego s� colocamos ensorflow en nuestro entorno virtual:

`sudo apt install libatlas3-base`
`pip3 install tensorflow`

Ahora comprobamos que ha quedado bien instalado:

`python`
`>>> import tensorflow as tf`
`>>> tf.__version__`

### Instalaci�n de OpenCV
Vamos a instalar los paquetes necesarios para hacer funcionar OpenCV en nustra tarjeta y luego instalamos la librer�a en nuestro entorno virtual ***(por ahora no dedespere si salen errores conla instalaci�n de los paquetes, luego lo vamos a arreglar)*** :

`sudo apt install libatlas3-base libwebp6 libtiff5 libjasper1 libilmbase12 libopenexr22 libilmbase12 libgstreamer1.0-0 libavcodec57 libavformat57 libavutil55 libswscale4 libqtgui4 libqt4-test libqtcore4`

`sudo pip3 install opencv-python`

Ahora comprobamos si OpenCV qued� bien instalado:

`python`
`>>> import cv2`
`>>> cv2.__version__`

Si le sale algo como lo de la im�gen de abajo, entonces vamos a instalar los paquetes necesarios para que funcione: 

[![](https://blog.piwheels.org/wp-content/uploads/2018/09/Screenshot-from-2018-09-27-17-51-11.png)](http://https://blog.piwheels.org/wp-content/uploads/2018/09/Screenshot-from-2018-09-27-17-51-11.png)

C�mbiese al siguiente directorio y busque un archivo de extensi�n .so:

`cd /usr/local/lib/python3.7/dist-packages/cv2`
`ls`

En mi caso l archivo ten�a nombre `cv2.cpython-37m-arm-linux-gnueabihf.so`, entonces corro `ldd` sobre ese archivo:

`ldd cv2.cpython-37m-arm-linux-gnueabihf.so`

Como respuesta usted va a ver en pantalla un mont�n de archivos con extensi�n .so, para algunos va a aparecer la direcci�n donde se encuentran (con estos no hay problema) y con otros va a mostrar que no fue encontrado "not found" (estos soon los que hay que arreglar), para filtrar solo los que no sen encontraron, usamos:

`ldd cv2.cpython-35m-arm-linux-gnueabihf.so | grep "not found"`

Para buscar en instalar los paquetes que faltan, vamos a instalar la herramienta `apt-file` y la actualizamos:

`sudo apt install apt-file`
`sudo apt-file update`

suponga que de los paquetes no encontrados hubo uno llamado `libhdf5_serial.so.100`, entonces busca su ubicaci�n o la de un sustituto as�:

`apt-file search libhdf5_serial.so.100`

A lo que va a obtener una respuesta similar a:

`libhdf5-100: /usr/lib/arm-linux-gnueabihf/libhdf5_serial.so.100`
`libhdf5-100: /usr/lib/arm-linux-gnueabihf/libhdf5_serial.so.100.0.1`

E instale alguno de los paquetes que aparecen antes de los dos puntos ":" de la siguiente manera:

`sudo apt install libhdf5-100`

**Este proceso debe realizarse con todos los paquetes que no fueron encontrados.**

Ahora vuelva a instalar OpenCV: 

`pip3 install opencv-python`

Ahora vuelva a comprobar si OpenCV ha quedado bien instalado.

