# LLM实战

在已有的基座模型的基础上开发

错误回复的三种可能原因

- 提示工程
- RAG
- 微调

RAG的原理

个人、企业知识库-Split-Embedding->向量数据库

向量数据库：向量化的文本数据

![image-20250415165516101](C:\Users\19736\AppData\Roaming\Typora\typora-user-images\image-20250415165516101.png)

模型微调可以算是小批量地训练模型

路线规划

![image-20250415170857517](C:\Users\19736\AppData\Roaming\Typora\typora-user-images\image-20250415170857517.png)

## Langchain

让LLM能够从私有数据库和文件中提取信息，并根据这些信息执行具体操作。

用于开发由语言模型驱动的应用程序的框架。

类似Springboot

**三个核心组件**

- compents：为LLM提供接口封装，模板提示，信息检索索引
- chains：组合组件来解决特定任务
- agents：与外部条件交互

其地位等价于数据库领域的JDBC

**六个组成部分**

- Models：LLMs

- Prompt Templates
- Chains
- Agents
- Embedding
- Indexes

**工作流程**

用户给出问题-在数据库中找到相关信息--和原始问题结合，给模型分析，得到一个答案-指导Agents采取行动

**LangSmith**

一个子产品，不是非有不可。

## 25.04.21

### 战术总结

Langchain有很多乱七八糟的库。这些暂时还不重要。

首先需要知道的，比较基础的是，使用别人的基座模型需要API_KEY，你可以使用os.environ提前在全局中定义好，也可以在调用model的时候显式写上。

### model

最简单的model.invoke()，它接受一个sequence对象，作为input，里面存放各种类型的msg对象。也可以额外添加configurable等参数。

理论上返回的对象需要用parser解析，但是貌似这个大字典也可以直接用键值对调用。

实际应用中，一般不会只给一个Msg，而是新建一个提示词模板，在模板中定义system和user msg，然后把模板类，model，parser三者用|连接，拼成一个chain调用，也是使用invoke()，这个invoke会接收一个字典格式（json格式）的东西，到模板中（键值一一对应），然后给model提问，然后parser解析，返回值。

熟悉的东西已经来了，这个json格式的东西实际上可以通过后端接受，用Python的后端库可以接受用户在网页上输入的信息，然后喂给大模型，再返回。

### 使用历史记录

chain的创建仍然同上

自顶而下地说，它现在是又把chain放到一个RunnableWithMessageHistory对象里面去了，而新建这个对象，除了chain以外，还需要一个input_messages_key，这个key和invoke时候接受的key是一样的，相当于接受用户输入到这个对象里。

除此之外，它还需要一个函数，这个函数能够调用ChatMessageHistory()，在每次问答后记录问答信息为一个ChatMessageHistory对象，存在一个地方。

而这个函数会隐式调用一个全局变量store，这个变量不必被复杂地封装起来，langchain会很智能地自己调用。

**至于返回流式数据**，访问上述RunnableWithMessageHistory对象的stream属性即可。

### 加载知识库

一个sequences，存储了很多document对象。

然后使用Chroma.from_documents() 将其向量化，就算存好了。

可以使用RunnableLambda对象将其存为一个检索器retriever

接着你就可以将retriever封装到chain里，作为'context' key的value，被调用。

当然，在你的提示词里也要做相关修正。

### 使用搜索引擎寻找信息

建立一个TavilySearchResults对象

使用代理，填入tools和model到chat_agent_executor对象中，然后调用，可以使用模板，也可以直接invoke

### 爬取网页信息作为知识库

使用WebBaseLoader对象，配合bs4解析网页信息，用.load方法获得docs对象

然后用RecursiveCharacterTextSplitter分割文本。

经过向量化之后存入store

然后用as_retriever()方法建立一个检索器。

然后和prompt, model拼成一个chain，可以直接调用。

**结合前文，现在要加入对问答历史的分析**

首先要修改prompt，告诉llm结合前文

然后在初始化的时候加入一个占位符chat_history备用。

然后写一个get_session_id的函数，history是被嵌在prompt的模板里的，而模板又是在子链，所以需要建成一个父链。

这里的流程需要再议。

#### LLM操纵数据库

使用SQLDatabase.from_uri()连接到数据库

然后使用create_sql_query_chain()把model和db做成一个链，但是要写数据，还是需要一个tool，用RunnablePassthrough和chain封装在一起，再跟别的链套在一起作一个新链，然后调用。

**另一种写法**

1. 配置model
2. 配置sql
3. SQLDatabaseToolkit().get_tools()
4. 写prompt
5. 创建代理整合model，tool，Prompt
6. .invoke()

## 4.24 战术总结x2

### **目前掌握的**

1. #### 带上历史记录问答

2. #### 搜索引擎

3. #### 直接爬取网页信息

4. #### 连接/操纵数据库

5. #### 构建向量数据库

6. #### 结构化文本存入数据库

7. #### 生成结构化的数据

8. #### 文本分类

9. #### 文本摘要

10. ### 以上的排列组合

Agent，基本是残废的，目前的水平最多就操纵个数据库，跟各类app的接口都没关系。

LLM，在能够采集到一定量数据的情况下，还只能到“囤积”个人知识库辅助问答的程度，还没到“训练”、“微调”和“蒸馏”之类的程度，当然，硬件可能也不支持。

## 25.08.07

在完成邮件后，需要对以上部分做一个综合运用的小demo，以确实理解和掌握LangChain。
