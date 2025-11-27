# 服务器端代码
# 目前不是真正的服务器端，只是一个简单的 Flask 应用
import flask

class DogServer:
    def __init__(self, memory_manager, cam_manager, speech_engine):
        self.memory_manager = memory_manager
        self.cam_manager = cam_manager
        self.speech_engine = speech_engine
        self.app = flask.Flask(__name__)
        self.app.route('/')(self.index)
        self.app.route('/frame')(self.get_frame)
        self.app.route('/history')(self.get_history)
        self.app.route('/medicine')(self.get_medicine)
        self.app.route('/medicine', methods=['POST'])(self.set_medicine)
    def index(self):
        return "RememberDog Server"
    def get_frame(self):
        """获取当前帧的副本"""
        frame = self.cam_manager.get_frame()
        if frame is not None:
            return flask.send_file(frame, mimetype='image/jpeg')
        return "No frame available", 404
    def get_history(self):
        """获取语音历史记录"""
        history = self.speech_engine.get_history()
        return flask.jsonify(history)
    def get_medicine(self):
        """获取当前设置的药品"""
        medicine = self.memory_manager.get_medicine()
        return flask.jsonify(medicine)
    def set_medicine(self):
        """设置当前药品"""
        data = flask.request.json
        medicine = data.get('medicine')
        if medicine:
            self.memory_manager.set_medicine(medicine)
            return flask.jsonify({"status": "success", "medicine": medicine})
        return flask.jsonify({"status": "error", "message": "No medicine provided"}), 400
    def run(self):
        self.app.run(host='0.0.0.0', port=5000)
    if __name__ == '__main__':
        server = DogServer(memory_manager, cam_manager, speech_engine)
        server.run()