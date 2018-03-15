import cv2
import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.camera import Camera
from kivy.graphics.texture import Texture

class CvCamera(Camera):
    def _camera_loaded(self, *largs):
        if kivy.platform=='android':
            self.texture = Texture.create(size=self.resolution,colorfmt='bgr')
            self.texture_size = list(self.texture.size)
        else:
            super(CvCamera, self)._camera_loaded()

    def on_tex(self, *l):
        if kivy.platform=='android':
            buf = self._camera.grab_frame()
            if not buf:
                return
            frame = self._camera.decode_frame(buf)
            buf = self.process_frame(frame)
            self.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        super(CvCamera, self).on_tex(*l)

    def process_frame(self,frame):
        return cv2.flip(frame,1).tostring()

kv = '''
BoxLayout:
    orientation: 'vertical'
    CvCamera:
        id: camera
        resolution: (640, 480)
        play: True
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
'''

class CamApp(App):
    __version__ = '1.0'
    def build(self):
        return Builder.load_string(kv)

if __name__ == '__main__':
    CamApp().run()
