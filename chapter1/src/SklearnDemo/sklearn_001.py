# ====特征处理，特征提取
"""
sklearn 数据集返回值介绍
 load和fetch返回的数据类型datasets.base.Bunch(字典格式)
    data:特征数据数组，是[n_samples*n_features]的二维numpy.ndarray数组
    target:标签数组，是n_samples的一维numpy.ndarray数组
    DESCR:数据描述
    feature_names:特征名，新闻数据，手写数字，回归数据集没有
    target_names:标签名
"""
# 数据集的划分api
"""
sklearn.model_selection.train_test_split(arrays,*options)
    x 数据集的特征值
    y 数据集的特征值
    test_size 测试集的大小，一般为float
    random_size 随机数种子，不同的种子会造成不同的随机采样结果。相同的种子采样结果相同
    return 训练集特征值，测试集特征值，训练集目标值，测试集目标值   --这个主意
"""
# 特征提取API
"""

字典特征提取:
  作用：对字典数据进行特征值化
   sklearn.feature_extraction.DictVectorizer(sparse=True,...)
   DictVectorizer.fit_transform(X)X:字典或者包含字典的迭代返回值：返回sparse矩阵
   DictVectorizer.inverse_transform(X)X:array 数组或者sparse矩阵返回值 ：转换之前数据格式
   DictVectorizer.get_feature_names()返回类别名称
文本特征提取
作用：对文本数据进行特征值化
sklearn.feature_extraction.text.CountVectorizer(stop_words=[]) 
返回词频矩阵
CountVectorizer.fit_transform(X) X:文本或者包含文本字符串的可迭代对象 返回值：返回sparse矩阵
CountVectorizer.inverse_transform(X) X:array 数组或者sparse矩阵返回值 ：转换之前数据格式
CountVectorizer.get_feature_names() 返回单词列表
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba

def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取鸢尾花数据集
    iris = load_iris()
    # print("鸢尾花数据集的返回值：\n", iris)
    # print("查看数据集描述：\n",iris.DESCR)
    # print("查看特征的名字：\n", iris["feature_names"])
    # print("查看数据集描述：\n", iris.data,iris.data.shape)

    # 数据集划分
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,test_size=0.2,random_state=22)
    print("训练集的特征值:\n",x_train,x_train.shape)
    return  None

def dict_demo():
    """
    # 字典特征提取
    :return:
    """
    data = [{"city":"北京","temperature":100},
            {"city":"上海","temperature":60},
            {"city":"广州","temperature":30}]
    # 1:实例化一个转换器类
    transfer = DictVectorizer(sparse=True)
    # 2：调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new.toarray(),type(data_new))
    print("字典特征名字:\n",transfer.get_feature_names())
    return  None

def text_demo():
    """
    # 英文本特征提取
    :return:
    """
    # 1:实例化一个转换器类
    data=["Life is short, i like like python","Life is too long, i dislike python"]
    transfer = CountVectorizer(stop_words=[])
    # 2：调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray(),type(data_new))
    print("文本特征名字:\n", transfer.get_feature_names())
    return None

def text_chinese_demo():
    """
    # 中文本特征提取,自动分词
    :return:
    """
    # 将中文文本分词
    data=["男生向女生表白的话：没有你，一分钟，",
          "太长友情的句子大全：时间冲不淡友谊的酒，",
          "距离拉不开思念的手心死的个性签名：我终生的等待，换不来你刹那的凝眸",
          "关于仿写句子选摘:",
          "爱国诗大全：早岁那知世事艰，中原北望气如山",
          "七夕节祝福语大全",
          "席慕容经典语录_席慕容经典爱情语录_席慕容语句",
          "感人的句子大全：把脸一直向着阳光，这样就不会见到阴影",
          "最美好的话大全：当我们爱这个世界时，才生活在这个世界上",
          "爱情的句子大全：爱情这件事情，从来不卑微",
          "关于心情不好的说说：哭给自己听，笑给别人看，这就是人生",
          "感慨的句子大全：得失是一个永恒的话题，得失也是成败的标尺"]
    chines_word=[]
    for send in data:
        chines_word.append(cut_word(send))

    # 1:实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一直"])
    # 2：调用fit_transform
    data_new = transfer.fit_transform(chines_word)
    print("data_new:\n", data_new.toarray(),type(data_new))
    print("文本特征名字:\n", transfer.get_feature_names())
    return None

def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征提取
    :return:
    """
    # 将中文文本分词
    data = ["男生向女生表白的话：没有你，一分钟，",
            "太长友情的句子大全：时间冲不淡友谊的酒，",
            "距离拉不开思念的手心死的个性签名：我终生的等待，换不来你刹那的凝眸",
            "关于仿写句子选摘:",
            "爱国诗大全：早岁那知世事艰，中原北望气如山",
            "七夕节祝福语大全",
            "席慕容经典语录_席慕容经典爱情语录_席慕容语句",
            "感人的句子大全：把脸一直向着阳光，这样就不会见到阴影",
            "最美好的话大全：当我们爱这个世界时，才生活在这个世界上",
            "爱情的句子大全：爱情这件事情，从来不卑微",
            "关于心情不好的说说：哭给自己听，笑给别人看，这就是人生",
            "感慨的句子大全：得失是一个永恒的话题，得失也是成败的标尺"]
    chines_word = []
    for send in data:
        chines_word.append(cut_word(send))

    # 1:实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一直"])
    # 2：调用fit_transform
    data_new = transfer.fit_transform(chines_word)
    print("data_new:\n", data_new.toarray(), type(data_new))
    print("文本特征名字:\n", transfer.get_feature_names())
    return None

def cut_word(text):
    """
    进行中文分词
    :param text:
    :return:
    """
    words =" ".join(list(jieba.cut(text)))
    return words
if __name__=="__main__":
    # 代码1：sklearn数据集使用
    # datasets_demo()
    # 字典特征转换
    # dict_demo()
    # 英文本特征转换
    # text_demo()
    # 中文本特征转换
    # text_chinese_demo()
    # 使用TF-IDF分词器
    tfidf_demo()