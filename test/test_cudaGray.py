import ctypes, os.path, platform
import unittest


class DLL:
    def __init__(self, path):
        self.dll = ctypes.cdll.LoadLibrary(path)

class Gray(DLL):
    def __init__(self, path):
        super().__init__(path)

    def call_cudaGray(self, input_path):
        # https://github.com/rfk/pyenchant/issues/45
        # TypeError: bytes or integer address expected instead of str i nstance
        input_path = ctypes.c_char_p(bytes(input_path, 'utf-8'))
        self.dll.test_cudaGray.argtype = [ctypes.c_char_p]
        self.dll.test_cudaGray.retypes = ctypes.c_bool
        try:
            result = self.dll.test_cudaGray(input_path)
        except:
            result = False
        return result


class TestGray(unittest.TestCase):
    def setUp(self):
        print(os.path.join(os.path.dirname(__file__), r'build/Release/test_accel_vision.dll'))
        self.test_dll = Gray(os.path.join(os.path.dirname(__file__), r'build/Release/test_accel_vision.dll'))
        self.image_path = "images/2048.jpg"

    
    # @unittest.expectedFailure
    def test_cudaGray(self):
        self.assertEqual(True, self.test_dll.call_cudaGray(self.image_path))


    def tearDown(self):
        self.test_dll = None
        self.image_path = ''


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGray)
    unittest.TextTestRunner(verbosity=2).run(suite)
