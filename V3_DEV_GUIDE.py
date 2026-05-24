"""
ComfyUI V3 节点开发参考
========================

本文件记录了将 V1 节点迁移到 V3 的关键经验，供后续节点开发快速参考。
基于 nano_banana.py 的实际迁移总结。

核心发现：V3 节点可以直接放入 V1 的 NODE_CLASS_MAPPINGS 中注册，
ComfyUI 通过 issubclass(obj_class, _ComfyNodeInternal) 自动识别并
调用 GET_NODE_INFO_V1() 生成前端所需的节点信息。无需 comfy_entrypoint。

=== 最小 V3 节点模板 ===

    from comfy_api.latest import io

    class MyNode(io.ComfyNode):

        @classmethod
        def define_schema(cls):
            return io.Schema(
                node_id="MyNode",           # 必须与 NODE_CLASS_MAPPINGS 的 key 一致
                display_name="我的节点",
                category="image/generation",
                inputs=[...],
                outputs=[io.Image.Output(display_name="输出")],
            )

        @classmethod
        def execute(cls, input1, input2, ...) -> io.NodeOutput:
            # 业务逻辑
            return io.NodeOutput(result)

=== V1 → V3 对照表 ===

    V1                              V3
    ─────────────────────────────────────────────────────
    INPUT_TYPES() classmethod       define_schema() → io.Schema
    RETURN_TYPES = ("IMAGE",)       outputs=[io.Image.Output()]
    RETURN_NAMES = ("输出",)        io.Image.Output(display_name="输出")
    FUNCTION = "generate"           固定为 execute
    CATEGORY = "xxx"                Schema(category="xxx")
    generate(self, ...)             execute(cls, ...) classmethod
    self.xxx 实例状态               模块级单例函数

=== DynamicCombo（动态联动下拉框）===

场景：一个 combo 的选项决定其他 combo 显示哪些值。

    io.DynamicCombo.Input("模型", options=[
        io.DynamicCombo.Option("选项A", [
            io.Combo.Input("子参数1", options=["x", "y"]),
            io.Combo.Input("子参数2", options=["1K", "2K"]),
        ]),
        io.DynamicCombo.Option("选项B", [
            io.Combo.Input("子参数1", options=["x", "y", "z", "w"]),
            io.Combo.Input("子参数2", options=["512px", "1K", "2K", "4K"]),
        ]),
    ])

execute 中接收为 dict：
    def execute(cls, 模型, ...):
        selected = 模型["模型"]      # "选项A" 或 "选项B"
        sub1 = 模型["子参数1"]       # 对应选项下的子输入值
        sub2 = 模型["子参数2"]

注意：dict 的 key 是 DynamicCombo.Input 的 id（"模型"），
子输入的 key 是各 Combo.Input 的 id。

=== Autogrow（自动增长输入槽）===

场景：用户连接一个槽后自动出现下一个，最多 N 个。

    io.Autogrow.Input("参考图",
        template=io.Autogrow.TemplatePrefix(
            input=io.Image.Input("img"),
            prefix="参考图",        # 生成 参考图0, 参考图1, ...
            min=0,                  # 最少显示几个槽
            max=9,                  # 最多几个槽
        ),
    )

execute 中接收为 dict（或 io.Autogrow.Type）：
    def execute(cls, 参考图=None, ...):
        if 参考图:
            for key, tensor in 参考图.items():
                # key = "参考图0", "参考图1", ...
                # tensor = IMAGE tensor 或 None

=== 实例状态处理 ===

V3 的 execute 是 classmethod，无法用 self。
用模块级单例替代：

    _client = None

    def _get_client():
        global _client
        if _client is None:
            _client = MyAPIClient()
        return _client

=== 注册方式（与 V1 共存）===

在 __init__.py 中照常注册，无需任何特殊处理：

    NODE_CLASS_MAPPINGS = {
        "MyV1Node": MyV1Node,       # V1 节点
        "MyV3Node": MyV3Node,       # V3 节点，自动识别
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "MyV1Node": "V1 节点",
        "MyV3Node": "V3 节点",      # 也可省略，V3 用 Schema.display_name
    }

=== 注意事项 ===

1. node_id 必须与 NODE_CLASS_MAPPINGS 的 key 完全一致
2. V3 execute 返回 io.NodeOutput(tensor)，不是 tuple
3. _wrap_generate_for_error_display 等 V1 包装器对 V3 无效
   （找不到 generate 方法会安全跳过）
4. V3 支持 async execute（直接加 async 即可）
5. 输入参数名必须与 Schema inputs 的 id 一致
6. DynamicCombo 的子输入在前端会随选项切换动态显示/隐藏
7. Autogrow 的 widget 输入会被强制为 force_input（仅连接，无控件）

=== 可用输入类型速查 ===

    io.String.Input(id, default="", multiline=False)
    io.Int.Input(id, default=0, min=0, max=N, step=1)
    io.Float.Input(id, default=0.0, min=0.0, max=N, step=0.01)
    io.Combo.Input(id, options=[...], default="...")
    io.Boolean.Input(id, default=False)
    io.Image.Input(id)
    io.Mask.Input(id)
    io.Latent.Input(id)
    io.DynamicCombo.Input(id, options=[DynamicCombo.Option(...)])
    io.Autogrow.Input(id, template=TemplatePrefix/TemplateNames)

=== 可用输出类型速查 ===

    io.Image.Output(display_name="...")
    io.String.Output(display_name="...")
    io.Int.Output()
    io.Float.Output()
    io.Latent.Output()
    io.Mask.Output()
"""
