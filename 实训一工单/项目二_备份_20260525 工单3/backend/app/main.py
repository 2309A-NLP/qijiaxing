# 【作用】这个文件是 FastAPI 应用的入口，整个后端的启动和路由挂载都在这里【原理】Python 把 main.py 当作模块的入口文件，FastAPI 从这里创建应用实例【逻辑】先导入依赖 → 配置日志 → 创建 app → 挂载中间件 → 挂载路由 → 定义接口 → 启动服务器
"""
backend/app/main.py — FastAPI应用入口

作用：FastAPI应用的启动入口，负责：
  1. 创建FastAPI应用实例
  2. 挂载中间件（CORS → Logging → RateLimit）
  3. 挂载路由（chat模块）
  4. 定义启动/关闭事件
  5. 定义健康检查接口

中间件执行顺序（从外到内）：
  请求进入 → CORS → Logging → RateLimit → 路由处理 → 响应返回
  添加顺序（代码顺序）：RateLimit → Logging → CORS
  实际执行：CORS最先处理请求，RateLimit最后处理请求

面试官可能问：
  Q: 为什么启动事件里不加载模型？
  A: 模型加载在首次请求时懒加载，不在启动事件做。原因是：
     如果模型加载失败（如文件损坏），启动事件抛异常会导致整个服务崩溃。
     而懒加载失败只影响那次请求，服务本身还能跑（返回降级结果）。
     这是Reranker异步预加载的设计延续。

  Q: 中间件的顺序为什么是CORS在最外层？
  A: CORS中间件需要在响应头加Access-Control-*字段。如果CORS在内层，
     被RateLimit拒绝的请求（429）不会经过CORS，浏览器会因为缺少
     跨域头而报错而不是显示429状态码。CORS在最外层确保所有响应
     （包括限流拒绝的）都有跨域头。

  Q: /health接口为什么不需要认证？
  A: 健康检查用于负载均衡器和K8s探活，它们没有Token。
     如果health需要认证，负载均衡器无法判断服务是否健康。
     这是业界惯例——health和metrics端点永远公开。

  Q: uvicorn.run()为什么不设置workers>1？
  A: workers>1会启动多个进程。模型加载是重量级操作，多进程
     会导致模型被加载多次（每进程一份），浪费显存。
     单进程+异步足以处理高并发（FastAPI异步非阻塞）。
"""

# 【作用】导入 Python 内置的 logging 日志模块，用来输出运行日志【原理】logging 是 Python 标准库，比 print 更灵活，可以控制日志级别和格式【逻辑】后面用 basicConfig 配置一次，然后通过 getLogger 拿到日志记录器
import logging

# 【作用】从 fastapi 框架中导入 FastAPI 类，用于创建应用实例【原理】FastAPI 是一个现代 Python Web 框架，基于 Starlette 和 Pydantic【逻辑】我们拿 FastAPI 类 new 出一个对象 app，它就是整个应用本体
from fastapi import FastAPI

# 【作用】导入跨域资源共享中间件，解决前后端不同域名/端口访问的问题【原理】浏览器的同源策略默认阻止跨域请求，CORSMiddleware 在响应头里加上 Access-Control-* 字段告诉浏览器允许跨域【逻辑】后面用 app.add_middleware 注册它
from fastapi.middleware.cors import CORSMiddleware

# 【作用】从当前包的 config 模块（即 app/config.py）导入 settings 对象【原理】settings 是一个 Pydantic Settings 实例，从环境变量或 .env 文件中读取配置【逻辑】后面所有的配置项（应用名、版本、端口等）都是从这个 settings 对象拿的
from .config import settings

# 【作用】导入聊天模块的路由器（router），并把它重命名为 chat_router 方便使用【原理】FastAPI 的 APIRouter 可以把一组路由单独写在另一个文件里，然后挂载到主应用上，实现模块化【逻辑】后面用 app.include_router 把聊天相关的接口挂进来
from .api.v1.chat import router as chat_router

# 【作用】导入日志中间件，用来记录每个请求的详细信息（方法、路径、耗时等）【原理】中间件会在请求进来和响应出去时拦截，LoggingMiddleware 在请求前后打日志【逻辑】中间件是按添加顺序执行的，日志中间件在 CORS 和限流之间
from .middleware.logging_middleware import LoggingMiddleware

# 【作用】导入限流中间件，防止单一 IP 短时间内大量请求打崩服务器【原理】RateLimitMiddleware 维护一个计数器，统计每个 IP 在时间窗口内的请求数，超过上限就返回 429 状态码【逻辑】限流中间件要放在最内层（最靠近路由），这样能尽早拒绝超限请求
from .middleware.rate_limit import RateLimitMiddleware

# 【作用】配置 logging 模块的全局设置：日志级别和输出格式【原理】settings.LOG_LEVEL 是一个字符串比如 "INFO"，getattr(logging, "INFO") 拿到 logging.INFO 这个常量（数字 20），basicConfig 只生效一次（第一次调用）【逻辑】format 字符串里的 %(asctime)s 是时间，%(levelname)s 是级别，%(name)s 是记录器名字，%(message)s 是真正的日志内容
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# 【作用】创建一个叫 __name__ 的日志记录器实例【原理】__name__ 在当前文件运行时是 "__main__"，被导入时是 "app.main"，这样日志里就能区分日志是哪个模块打印的【逻辑】后面代码里就都用 logger.info()、logger.warning() 等方法来写日志了
logger = logging.getLogger(__name__)

# 【作用】创建 FastAPI 应用实例，并设置标题、版本、描述和文档地址【原理】FastAPI(__init__) 接收各种元数据参数，title/version/description 会被显示在自动生成的 API 文档页面（/docs 和 /redoc）上【逻辑】各个参数都从 settings 配置对象读取，只有 description 是硬编码的中文描述
app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, description="招股说明书智能问答系统API", docs_url="/docs", redoc_url="/redoc")

# 中间件：CORS(最外层) → Logging → RateLimit(最内层)

# 【作用】添加限流中间件，设置为 60 秒内最多 100 个请求【原理】app.add_middleware() 把中间件按"后添加先执行"的顺序压入栈，RateLimitMiddleware 放在最前面（所以最早执行），在每个请求到达路由之前先检查是否超限【逻辑】max_requests=100 表示窗口内最多 100 次，window_seconds=60 表示窗口是 60 秒
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

# 【作用】添加请求日志中间件，记录每一次 HTTP 请求的详细信息【原理】LoggingMiddleware 在请求到来时记录方法、路径、客户端 IP，在响应返回时记录耗时和状态码【逻辑】这个中间件在限流之后、CORS 之前执行，这样日志既能记录被限流的请求，也能记录正常请求
app.add_middleware(LoggingMiddleware)

# 【作用】添加 CORS 跨域中间件，允许所有来源的跨域请求【原理】allow_origins=["*"] 表示允许任意域名访问，allow_credentials=True 允许携带 Cookie 等凭证，allow_methods=["*"] 允许所有 HTTP 方法，allow_headers=["*"] 允许所有请求头【逻辑】CORS 中间件在最外层（最后添加），所以它最先处理请求，在加上跨域响应头之后才交给日志和限流中间件
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 【作用】把聊天功能的路由器挂载到主应用上，所有接口前面都加 /api/v1 前缀【原理】app.include_router() 会把 router 里定义的所有路由注册到 app 上，prefix="/api/v1" 表示聊天接口的路径是 /api/v1/chat 之类【逻辑】tags=["v1"] 用于在 /docs 文档页面上分组显示，同一个 tags 的路由归到一起
app.include_router(chat_router, prefix="/api/v1", tags=["v1"])


# 【作用】用装饰器定义一个 GET 请求的 /health 健康检查接口【原理】@app.get("/health") 相当于 app.add_route("/health", health_check, methods=["GET"])，FastAPI 自动把函数返回值序列化为 JSON 响应【逻辑】这个接口不依赖数据库或其他服务，纯粹用来检查服务是否活着
@app.get("/health")
# 【作用】定义异步函数 health_check，这是 /health 接口的处理函数【原理】async def 声明异步函数，FastAPI 在协程中运行它，不会阻塞其他请求的处理【逻辑】函数直接返回一个字典，FastAPI 自动把它转成 JSON 格式的 HTTP 响应
async def health_check():
    # 【作用】返回一个包含服务状态、版本号和运行环境的 JSON 对象【原理】FastAPI 会把 dict 自动序列化成 JSON，设置 Content-Type: application/json【逻辑】status 固定返回 "healthy"，version 和 environment 从配置对象 settings 读取，方便运维人员判断当前是哪个版本在运行
    return {"status": "healthy", "version": settings.APP_VERSION, "environment": settings.ENV}


# 【作用】用装饰器注册一个"启动时执行"的事件处理函数【原理】@app.on_event("startup") 告诉 FastAPI 在应用启动完成后、开始接受请求之前，执行下面这个函数【逻辑】常用于初始化数据库连接、加载模型、预热缓存等一次性操作
@app.on_event("startup")
# 【作用】定义启动事件的处理函数，应用启动时自动调用【原理】async def 表示这是异步函数，FastAPI 在事件循环中 await 它【逻辑】这里只是打印两行日志，告知开发者服务已经成功启动并给出访问地址
async def startup_event():
    # 【作用】用 logger 输出服务启动成功的消息，包含应用名和版本号【原理】f-string 会把 {} 里的表达式计算结果嵌入字符串，logger.info() 按前面 basicConfig 配置的格式输出到控制台【逻辑】🚀 表情让日志更容易被肉眼找到，方便开发时确认服务是否启动
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} 启动")
    # 【作用】输出当前运行环境和 API 文档地址，方便开发者直接点击访问【原理】settings.HOST 和 settings.PORT 是配置对象里的主机地址和端口号【逻辑】把文档链接打印出来，开发者可以直接复制链接去浏览器打开 Swagger 文档页面
    logger.info(f"环境: {settings.ENV} | 文档: http://{settings.HOST}:{settings.PORT}/docs")


# 【作用】用装饰器注册一个"关闭时执行"的事件处理函数【原理】@app.on_event("shutdown") 告诉 FastAPI 在应用收到关闭信号（比如 Ctrl+C）后、进程退出前，执行下面这个函数【逻辑】常用于关闭数据库连接、释放资源、保存状态等清理工作
@app.on_event("shutdown")
# 【作用】定义关闭事件的处理函数，应用关闭时自动调用【原理】和 startup 一样是异步函数，在事件循环中执行【逻辑】这里只是打印一条日志告知开发者服务已关闭
async def shutdown_event():
    # 【作用】输出服务关闭的日志【原理】logger.info() 按配置的格式打印到控制台【逻辑】👋 表情表示再见，让日志更友好
    logger.info("👋 服务关闭")


# 【作用】判断当前文件是否被当作主程序直接运行（而不是被 import 导入）【原理】当 Python 执行 python main.py 时，__name__ 变量的值是 "__main__"；如果这个文件被别的文件 import，__name__ 就是模块名比如 "app.main"【逻辑】只有直接运行 main.py 时才启动 Uvicorn 服务器，被导入时不会自动启动
if __name__ == "__main__":
    # 【作用】导入 Uvicorn——一个 ASGI 服务器，用来运行 FastAPI 应用【原理】Uvicorn 是基于 uvloop 和 httptools 的高性能 ASGI 服务器，FastAPI 是基于 ASGI 规范的框架，所以需要 Uvicorn 来驱动【逻辑】把 import 写在 if 内部，这样只有直接运行时才导入 Uvicorn，被导入时就不会额外加载
    import uvicorn
    # 【作用】启动 Uvicorn 服务器，加载 FastAPI 应用并监听指定地址和端口【原理】"app.main:app" 表示在 app/main.py 文件中找到名为 app 的 FastAPI 实例，host 和 port 指定监听地址，workers=1 表示单进程，reload=settings.DEBUG 表示调试模式时自动重载代码变化【逻辑】host 和 port 从 settings 读取，这样不用改代码就能通过环境变量控制服务器地址；reload 在调试时开启方便开发，生产环境关闭提高性能
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, workers=1, reload=settings.DEBUG)
