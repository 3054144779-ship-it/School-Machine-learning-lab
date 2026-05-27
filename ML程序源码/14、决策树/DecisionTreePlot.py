import matplotlib.pyplot as plt

"""
定义文本框和箭头格式[sawtooth:波浪方框,round4:矩形方框,fc:字体颜色的深浅,0.1~0.9依次变浅]
"""
decision_node = dict(boxstyle = 'sawtooth', fc = '0.8')
leaf_node = dict(boxstyle = 'round4', fc = '0.8')
arrow_args = dict(arrowstyle = '<-')

def get_number_leaves(my_tree):
    number_leaves = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    """
    根节点开始遍历
    """
    for key in second_dict.keys():
        """
        判断子节点是否为dict,不是+1
        """
        if type(second_dict[key]) is dict:
            number_leaves += get_number_leaves(second_dict[key])
        else:
            number_leaves += 1
    return number_leaves

def get_tree_depth(my_tree):
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    """
    根节点开始遍历
    """
    for key in second_dict.keys():
        """
        判断子节点是不是dict,求分支的深度
        """
        if type(second_dict[key]) is dict:
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        max_depth = max(max_depth, this_depth)
    return max_depth

def plot_node(node_txt, center_pt, parent_pt, node_type):
    # pass
    create_plot.sp.annotate(node_txt, xy = parent_pt, xycoords = 'axes fraction', xytext = center_pt, textcoords = 'axes fraction', va = 'center', ha = 'center', bbox = node_type, arrowprops = arrow_args)

def plot_mid_text(center_point, parent_point, str_txt):
    # pass
    x_mid = (parent_point[0] - center_point[0]) / 2 + center_point[0]
    y_mid = (parent_point[1] - center_point[1]) / 2 + center_point[1]
    create_plot.sp.text(x_mid, y_mid, str_txt, va = 'center', ha = 'center', rotation = 30)

def plot_tree(my_tree, parent_point, node_txt):
    # pass
    number_leaves = get_number_leaves(my_tree)  # 获取叶子节点的数量
    """
    找出第1个中心点的位置,然后与parent_point定点进行划线,并打印输入对应的文字
    """
    center_point = (plot_tree.x_offset + (1 + number_leaves) / 2 / plot_tree.total_width, plot_tree.y_offset)
    plot_mid_text(center_point, parent_point, node_txt)
    first_str = list(my_tree.keys())[0]
    plot_node(first_str, center_point, parent_point, decision_node) # 可视化Node分支点
    second_dict = my_tree[first_str]    # 根节点的值
    plot_tree.y_offset = plot_tree.y_offset - 1 / (plot_tree.total_depth)   # y值 = 最高点 - 层数的高度[第二个节点位置]
    for key in second_dict.keys():
        if type(second_dict[key]) is dict:  # 判断该节点是否是Node节点
            plot_tree(second_dict[key], center_point, str(key)) # 如果是就递归调用
        else:
            plot_tree.x_offset = plot_tree.x_offset + 1 / plot_tree.total_width # 如果不是,就在原来节点一半的地方找到节点的坐标
            plot_node(second_dict[key], (plot_tree.x_offset, plot_tree.y_offset), center_point, leaf_node)  # 可视化该节点位置
            plot_mid_text((plot_tree.x_offset, plot_tree.y_offset), center_point, str(key)) # 并打印输入对应的文字
    plot_tree.y_offset = plot_tree.y_offset + 1 / (plot_tree.total_depth)


def create_plot(input_tree):
    """
    创建一个figure的模板
    """
    figure = plt.figure(1, facecolor = 'green')
    figure.clf()
    sp_props = dict(xticks = [], yticks = [])
    create_plot.sp = plt.subplot(111, frameon = False, **sp_props)  # 表示创建一个1行1列的图,create_plot.sp为第1个子图
    plot_tree.total_width = float(get_number_leaves(input_tree))
    plot_tree.total_depth = float(get_tree_depth(input_tree))
    """
    半个节点的长度
    """
    plot_tree.x_offset = -0.5 / (plot_tree.total_width)
    plot_tree.y_offset = 1.0
    plot_tree(input_tree, (0.5, 1.0), '')
    plt.show()