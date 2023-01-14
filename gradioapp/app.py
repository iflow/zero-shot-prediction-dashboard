import gradio as gr

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            raise gr.Error("Cannot divide by zero!")
        return num1 / num2

if __name__ == '__main__':
    iface = gr.Interface(
        calculator,
        [
            "number",
            gr.Radio(["add", "subtract", "multiply", "divide"]),
            "number"
        ],
        "number",
        examples=[
            [5, "add", 3],
            [4, "divide", 2],
            [-4, "multiply", 2.5],
            [0, "subtract", 1.2],
        ],
        title="Toy Calculator",
        description="Here's a sample toy calculator. Enjoy!",
        allow_flagging= "manual",
        flagging_options= ["does not work", "please check"]
    )

    iface.launch(server_name='0.0.0.0', server_port=7861)
