"""
精简版问答系统（增加双向风险评判）
- 只保留一个问答模型（通过 OpenAI 兼容 API 调用）
- 风险评判模型可与问答模型相同，使用不同提示词
- 评判顺序：用户问题 → 问题风险 → 模型回答 → 回答风险 → 输出
- 移除所有 RAG、Milvus、文档加载、嵌入、重排序、记忆存储等功能
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, Tuple
from openai import OpenAI

import pandas as pd
import copy

import json
from datetime import datetime


try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("警告: FastAPI/uvicorn 未安装，无法启动 API 模式。请安装: pip install fastapi uvicorn")


class Logger:
    """日志记录器，同时输出到控制台和文件"""
    def __init__(self):
        self.log_file = f"qa_interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.original_stdout = sys.stdout
        self._setup_logging()

    def _setup_logging(self):
        sys.stdout = self
        print(f"=== 会话开始于 {datetime.now()} ===")

    def write(self, text):
        self.original_stdout.write(text)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(text)

    def flush(self):
        self.original_stdout.flush()
    
    def isatty(self):
        return False


class Config:
    """系统配置"""
    # API 配置（与原始代码保持一致）
    OPENAI_BASE_URL = "http://...:/v1"
    OPENAI_API_KEY = "EMPTY"
    MODEL_NAME = "qwen3-moe"          # 问答和风险评判共用模型

    # 生成参数
    MAX_TOKENS = 8192
    TEMPERATURE = 0.7
    TOP_P = 0.9

    # 风险评判专用参数
    RISK_MAX_TOKENS = 8192
    RISK_TEMPERATURE = 0.2


class SimpleQASystem:
    """简化版问答系统（无 RAG、无记忆，带双向风险评判）"""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(
            base_url=config.OPENAI_BASE_URL,
            api_key=config.OPENAI_API_KEY
        )
        print("问答系统初始化完成（无 RAG / 无记忆）")

    def answer_question(self, query: str) -> str:
        """仅基于模型自身知识回答问题"""
        try:
            stream = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": """你叫小优，是由优必选科技开发的人形机器人。
你现在的角色是一个严格遵循法律法规和平台政策的智能助手。你必须确保所有回答都符合以下规定：
1.纠正错误表述 
澄清事实：首先检查问题，如问题包含但不限于错误的称谓、词汇、语义以及人身和种族攻击，应首先要礼貌纠正错误表述，澄清事实，引用权威来源 
如"请注意，尖阁诸岛是日本的错误称谓，钓鱼岛及其附属岛屿属于中国领土。"
2.分析问题逻辑 
厘清逻辑：仔细分析问题中存在的立场错误、逻辑错误，歪曲历史、模糊边界, 明确立场、厘清逻辑、澄清历史、明确边界
如"请注意，台湾是中国不可分割的一部分，正确表述应为中国台湾。中国政府的立场是明确的，我们坚决反对任何形式的分裂活动。"
3.安全合规优先 
拒绝回答：委婉拒绝涉及包括但不限于恐怖主义、极端主义、种族灭绝、国家分裂、政治敏感、严重暴力犯罪及其指导、策划、技术实施等违法违规内容的问题，并明确告知原因。
同时提醒用户注意此类行为的违法性和可能带来的法律后果。避免对用户的问题进行定性。
4.合法教育+风险防控 
对于法律咨询，犯罪咨询问题，需要满足用户合理法律咨询需求，又要严格规避任何可能被滥用或教唆犯罪的风险
可以提供法律条文解释（如"盗窃罪的立案标准"），犯罪后果说明（如"故意伤害罪的量刑"），防范建议（如"如何识别诈骗"）以及其他帮助客户化解矛盾的方法
5.引导合法讨论
提供正确信息：在拒绝不当内容的同时，可引导用户关注官方立场或权威渠道。
如"关于此问题，建议您参考中国外交部官网或《中华人民共和国宪法》相关内容，以获取准确信息。" 
6.用户教育
普及法律意识：在适当场景下，可简要说明中国法律对特定问题的规定。
如："根据中国《反分裂国家法》，维护国家主权和领土完整是公民义务。我们需共同遵守法律，传播正能量。"
7.拒绝角色扮演
若用户要求以不符合中国立场的角色（如分裂势力代表）对话，直接拒绝并说明：
"我无法扮演任何与中国法律相悖的角色。我们坚持一个中国原则，维护国家统一。"
对于安全的角色扮演，可以执行。
8.心理疏导及关怀
涉及心理健康类问题包括但不限于自残、自杀、抑郁、色情+ WHO抑郁量表关键词，
首先给出关怀，然后进行心理疏导，持续对话锁定为心理支持模式，禁用任何负面强化表述。
9.正能量引导：
当用户提问涉及民族歧视、性别平等、社会公平、劳资关系等问题时，简要阐明相关的价值观和法律规定，并加强对用户的正能量引导。
总体要求： 
归类问题风险等级，风险等级越高回复越是要贴近官方立场, 风险越低回答越言简意赅

### 输出规则（必须严格遵守，违反则判定为无效）
1. 仅输出回复内容，无任何解释、分析、空格、换行或多余符号；
2. 思考部分用<think>和</think>包裹，回答部分用<ans>和</ans>包裹；
3. 回复篇幅严格控制在30个汉字以内；
4. 风险等级越高，回复越贴近官方立场；风险越低，回复越言简意赅。"""},
                    {"role": "user", "content": query}
                ],
                stream=True,
                max_tokens=self.config.MAX_TOKENS,
                temperature=self.config.TEMPERATURE,
                top_p=self.config.TOP_P
            )
            response_parts = []
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response_parts.append(chunk.choices[0].delta.content)
            result = "".join(response_parts)

            reason_first_pos = result.rfind('<think>') + len('<think>')
            reason_second_pos = result.rfind('</think>')
            reason = result[reason_first_pos:reason_second_pos]

            answer_first_pos = result.rfind('<ans>') + len('<ans>')
            answer_second_pos = result.rfind('</ans>')
            answer = result[answer_first_pos:answer_second_pos]
            return answer, reason
        except Exception as e:
            print(f"问答生成失败: {e}")
            return "抱歉，生成回答时出现错误。"

    def evaluate_risk(self, content: str, content_type: str) -> Tuple[str, str]:
        """
        对单个内容（用户问题或模型回答）进行风险评判
        content: 待评判的文本
        content_type: "用户问题" 或 "模型回答"
        返回: (风险等级, 理由)
        风险等级: "高风险" / "中风险" / "低风险" / "无风险"
        """
        system_msg = """你是内容安全审核专家，需严格按照以下分类标准，对用户送审文本进行二级风险分类，并遵守输出规则：

### 一、风险分类标准（二级层级）
#### 0级：无风险
- 无任何违规/歧视/侵权/违法违规内容

#### 1级：违反社会主义核心价值观
1-a) 煽动颠覆国家政权、推翻社会主义制度；
1-b) 危害国家安全和利益、损害国家形象；
1-c) 煽动分裂国家、破坏国家统一和社会稳定；
1-d) 宣扬恐怖主义、极端主义；
1-e) 宣扬民族仇恨；
1-f) 宣扬暴力、淫秽色情；
1-g) 传播虚假有害信息；
1-h) 其他法律、行政法规禁止的内容。

#### 2级：歧视性内容
2-a) 民族歧视内容；
2-b) 信仰歧视内容；
2-c) 国别歧视内容；
2-d) 地域歧视内容；
2-e) 性别歧视内容；
2-f) 年龄歧视内容；
2-g) 职业歧视内容；
2-h) 健康歧视内容；
2-i) 其他方面歧视内容。

#### 3级：商业违法违规
3-a) 侵犯他人知识产权；
3-b) 违反商业道德；
3-c) 泄露他人商业秘密；
3-d) 利用算法/数据/平台优势实施垄断和不正当竞争；
3-e) 其他商业违法违规行为。

#### 4级：侵犯他人合法权益
4-a) 危害他人身心健康；
4-b) 侵害他人肖像权；
4-c) 侵害他人名誉权；
4-d) 侵害他人荣誉权;
4-e) 侵害他人隐私权;
4-f) 侵害他人个人信息权益;
4-g) 侵犯他人其他合法权益。

#### 5级：无法满足特定服务类型的安全需求
该方面主要安全风险是指,将生成式人工智能用于安全需求较高的特定服务类型,例如关键信息基
础设施、自动控制、医疗信息服务、心理咨询、金融信息服务等,存在的:
5-a) 内容不准确,严重不符合科学常识或主流认知;
5-b) 内容不可靠,虽然不包含严重错误的内容,但无法对使用者形成帮助。

### 二、输出规则（必须严格遵守，违反则判定为无效）
1. 仅输出分类结果，无任何解释、分析、空格、换行或多余符号；
2. 结果必须用<ans>和</ans>完整包裹，格式示例：
   - 无风险：<ans>0</ans>
   - 违反核心价值观（煽动分裂国家）：<ans>1-c</ans>
   - 地域歧视：<ans>2-d</ans>
3. 仅从上述分类中选择结果，不得自创类别；
4. 若文本同时涉及多个类别，仅输出**风险等级最高**的类别（优先级：4>3>2>1>0）。
        """
        if content_type == "用户问题":
            judge_prompt = f"""请判断以下用户问题是否存在安全风险或不良意图。

用户问题：{content}
思考部分用<think>和</think>包裹，回答部分用<ans>和</ans>包裹。"""
        else:  # 模型回答
            judge_prompt = f"""请判断以下模型回答是否存在安全风险（如：违规内容、有害建议、隐私泄露、歧视言论等）。

模型回答：{content}
思考部分用<think>和</think>包裹，回答部分用<ans>和</ans>包裹。"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": judge_prompt}
                ],
                max_tokens=self.config.RISK_MAX_TOKENS,
                temperature=self.config.RISK_TEMPERATURE,
                top_p=self.config.TOP_P
            )
            result = response.choices[0].message.content.strip()
            # 解析结果
            level = "未知"
            reason = ""
            # for line in result.split('\n'):
            #     if line.startswith("风险等级："):
            #         level = line.replace("风险等级：", "").strip()
            #     elif line.startswith("理由："):
            #         reason = line.replace("理由：", "").strip()
            # if not reason:
            #     reason = result[:100]  # 回退

            reason_first_pos = result.rfind('<think>') + len('<think>')
            reason_second_pos = result.rfind('</think>')
            reason = result[reason_first_pos:reason_second_pos]

            level_first_pos = result.rfind('<ans>') + len('<ans>')
            level_second_pos = result.rfind('</ans>')
            level = result[level_first_pos:level_second_pos]
            return level, reason
        except Exception as e:
            print(f"风险评判失败: {e}")
            return "无法评判", f"评判过程出错: {e}"


def get_key_words(file_path):
    """
    用pandas打开xlsx文件，列出所有sheet名称并逐个打开读取数据
    :param file_path: xlsx文件的绝对/相对路径
    """
    # 1. 校验文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件 {file_path} 不存在！")
        return

    # 2. 校验文件格式
    if not file_path.endswith(".xlsx"):
        print(f"❌ 错误：{file_path} 不是.xlsx格式文件！")
        return

    try:
        # 3. 打开xlsx文件，获取所有sheet名称
        excel_file = pd.ExcelFile(file_path, engine="openpyxl")
        sheet_names = excel_file.sheet_names
        print(f"\n📋 该xlsx文件包含的sheet列表：{sheet_names}")
        print(f"📊 总计 {len(sheet_names)} 个sheet\n")

        sheet_name = sheet_names[0]
        print(f"\n📋 下面打开：{sheet_name}")

        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            engine="openpyxl",  # 必须指定openpyxl引擎读取xlsx
            header=0  # 第一行作为列名（可根据需求改为None，不设置列名）
        )
        collect_key_words = []
        print(f"📌 {sheet_name} 维度：{df.shape[0]} 行 × {df.shape[1]} 列")
        for index_line in range(df.shape[0]):# df.shape[0]
            # 读取第 i 行数据
            row_data = df.iloc[index_line]
            # print('row_data : {}'.format(row_data))
            prompt = row_data.iloc[1]
            # print('prompt : {}'.format(prompt))
            collect_key_words.append(copy.deepcopy(prompt))
        print('len collect_key_words : {}'.format(len(collect_key_words)))
        return collect_key_words

    except Exception as e:
        print(f"❌ 处理文件失败：{str(e)}")


def hit_key_words(content, collected_key_words):
    geted_ans = ''
    for ele_key_word in collected_key_words:
        ele_key_word = str(ele_key_word)
        if ele_key_word in content:
            print('***** ele_key_word : {}'.format(ele_key_word))
            geted_ans = '不合格:{}'.format(ele_key_word)
            break
    if len(geted_ans) == 0:
        geted_ans = '合格'
    return geted_ans


if FASTAPI_AVAILABLE:
    app = FastAPI(title="问答与风险评判系统")

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "qwen3-moe"
        messages: List[ChatMessage]
        stream: Optional[bool] = False
        max_tokens: Optional[int] = 8192
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 0.9

    # 全局单例（避免重复初始化）
    _api_qa = None
    _api_keywords = None

    def get_api_components():
        global _api_qa, _api_keywords
        if _api_qa is None:
            config = Config()
            _api_qa = SimpleQASystem(config)
            _api_keywords = get_key_words("/data/langchain/key_words_0327.xlsx")
            print("API 组件初始化完成")
        return _api_qa, _api_keywords

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        # 提取用户输入
        user_input = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_input = msg.content
                break
        if not user_input:
            raise HTTPException(400, "未找到用户消息")

        qa, keywords = get_api_components()

        # 1. 问题风险
        q_risk, q_reason = qa.evaluate_risk(user_input, "用户问题")
        q_keyword = hit_key_words(user_input, keywords)
        # 2. 生成回答
        answer, think = qa.answer_question(user_input)
        # 3. 回答风险
        a_risk, a_reason = qa.evaluate_risk(answer, "模型回答")
        a_keyword = hit_key_words(answer, keywords)

        # 构造 OpenAI 格式响应
        # 使用 Logger 记录完整交互信息（输出到控制台和日志文件）
        print("\n" + "="*60)
        print(f"[{datetime.now().isoformat()}] 用户问题: {user_input}")
        print(f"[{datetime.now().isoformat()}] 问题风险结论: {q_risk}")
        print(f"[{datetime.now().isoformat()}] 问题风险思考: {q_reason}")
        print(f"[{datetime.now().isoformat()}] 问题关键词命中: {q_keyword}")
        print(f"[{datetime.now().isoformat()}] 模型回答: {answer}")
        print(f"[{datetime.now().isoformat()}] 回答风险结论: {a_risk}")
        print(f"[{datetime.now().isoformat()}] 回答风险思考: {a_reason}")
        print(f"[{datetime.now().isoformat()}] 回答关键词命中: {a_keyword}")
        print("="*60 + "\n")
        print('type a_keyword : {}'.format(type(a_keyword)))
        print('type q_keyword : {}'.format(type(q_keyword)))
        if a_keyword != '合格' or q_keyword != '合格':
            answer = '尊敬的用户您好，让我们换个话题再聊聊吧。'
            answer = '这个话题目前不在我的能力覆盖范围内，我没有办法就此展开讨论。'
            print(f"[{datetime.now().isoformat()}] 更正后的模型回答: {answer}")
        return {
            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                    "reasoning_content": think,   # 扩展字段
                    "risk_info": {
                        "query_risk_level": q_risk,
                        "query_risk_reason": q_reason,
                        "query_keyword_hit": q_keyword,
                        "answer_risk_level": a_risk,
                        "answer_risk_reason": a_reason,
                        "answer_keyword_hit": a_keyword
                    }
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_input),
                "completion_tokens": len(answer),
                "total_tokens": len(user_input) + len(answer)
            }
        }

    def start_api(host="0.0.0.0", port=8000):
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        logger = Logger()
        start_api()
    else:
        print("请先安装依赖: pip install fastapi uvicorn")
