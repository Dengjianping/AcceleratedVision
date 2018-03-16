import ctypes, os.path, platform
import unittest


class DLL:
    def __init__(self, path):
        self.dll = ctypes.cdll.LoadLibrary(path)


class Theshold(DLL):
    def __init__(self, path):
        super().__init__(path)


    def test_cudaThreshold(self, input_path, thresh, max_value):
        # https://github.com/rfk/pyenchant/issues/45
        # TypeError: bytes or integer address expected instead of str i nstance
        input_path = ctypes.c_char_p(bytes(input_path, 'utf-8'))
        thresh = ctypes.c_char(thresh)
        max_value = ctypes.c_char(max_value)
        self.dll.test_cudaThreshold.argtype = [ctypes.c_char_p, ctypes.c_char, ctypes.c_char]
        self.dll.test_cudaThreshold.retypes = ctypes.c_bool
        result = self.dll.test_cudaThreshold(input_path, thresh, max_value)
        return result


class TestTheshold(unittest.TestCase):
    def setUp(self):
        self.test_dll = Theshold(os.path.join(os.path.dirname(__file__), r'build/Release/test_accel_vision.dll'))
        self.image_path = "images/2048.jpg"

    
    def test_cudaThreshold(self):
        thresh, max_value = 50, 200
        self.assertEqual(True, self.test_dll.test_cudaThreshold(self.image_path, thresh, max_value))


    @unittest.expectedFailure
    def test_cudaThreshold(self):
        thresh, max_value = -50, 2000
        self.assertEqual(False, self.test_dll.test_cudaThreshold(self.image_path, thresh, max_value))


    def tearDown(self):
        self.test_dll = None


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTheshold)
    unittest.TextTestRunner(verbosity=2).run(suite)