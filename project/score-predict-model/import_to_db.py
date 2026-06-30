"""将 Score_dataset.xlsx 的7个Sheet数据导入 MySQL t_student_history 表"""
import pandas as pd
import pymysql
import os

BASE_PATH = os.path.dirname(__file__)
EXCEL_PATH = os.path.join(BASE_PATH, "data", "Score_dataset.xlsx")

# MySQL 连接配置（与 application.properties 一致）
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root123456",
    "database": "student_predict",
    "charset": "utf8mb4",
}

# Excel 列名 → 数据库列名 映射
COLUMN_MAP = {
    "学生姓名": "student_name",
    "线上_平时成绩": None,       # 共线特征，不导入
    "线上_期中测试": None,       # 共线特征，不导入
    "线上_期末考试": None,       # 共线特征，不导入
    "线上总成绩": "online_total",
    "线下_互动": "interaction",
    "线下_期末考试": None,       # 不在特征列中，跳过
    "线下总成绩": None,          # 不在特征列中，跳过
    "综合_平时成绩": "comprehensive_regular",
    "期末总成绩": "final_total",
    "平时成绩": "regular_score",
    "期末成绩": "final_score",
}


def main():
    xl = pd.ExcelFile(EXCEL_PATH)
    print(f"Excel 共 {len(xl.sheet_names)} 个Sheet: {xl.sheet_names}")

    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 清空旧数据
    cursor.execute("TRUNCATE TABLE t_student_history")
    print("已清空旧数据")

    total_rows = 0
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name)

        # 重命名列为数据库字段名
        rename_map = {k: v for k, v in COLUMN_MAP.items() if v is not None and k in df.columns}
        df_import = df[list(rename_map.keys())].rename(columns=rename_map)

        # 填充 NaN → None
        df_import = df_import.astype(object)
        df_import = df_import.where(df_import.notna(), None)

        columns = ", ".join(df_import.columns)
        placeholders = ", ".join(["%s"] * len(df_import.columns))
        sql = f"INSERT INTO t_student_history ({columns}) VALUES ({placeholders})"

        # 逐行转换，确保 NaN 全部替换为 None
        rows = []
        for _, row in df_import.iterrows():
            vals = tuple(None if pd.isna(v) else v for v in row.values)
            rows.append(vals)
        cursor.executemany(sql, rows)
        conn.commit()

        total_rows += len(rows)
        print(f"  {sheet_name}: 导入 {len(rows)} 条")

    cursor.close()
    conn.close()
    print(f"\n导入完成，共 {total_rows} 条记录。")


if __name__ == "__main__":
    main()
