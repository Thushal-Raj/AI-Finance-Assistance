from flask import Flask, request, redirect, render_template_string
import sqlite3
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression

app = Flask(__name__)

# ============================================================
# AI EXPENSE CATEGORY MODEL
# ============================================================

training_texts = [
    "swiggy biryani",
    "restaurant dinner",
    "pizza",
    "lunch food",
    "breakfast",
    "zomato food",
    "groceries",
    "chicken rice",

    "uber ride",
    "ola cab",
    "bus ticket",
    "petrol",
    "fuel",
    "metro ticket",
    "auto fare",

    "amazon shopping",
    "clothes",
    "shoes",
    "shopping mall",
    "online purchase",
    "shirt",

    "electricity bill",
    "water bill",
    "internet bill",
    "mobile recharge",
    "phone bill",

    "movie ticket",
    "netflix",
    "game",
    "concert",
    "entertainment",

    "medicine",
    "doctor",
    "hospital",
    "pharmacy",

    "college fees",
    "books",
    "course",
    "education",
    "exam fee",
    "tuition"
]

training_categories = (
    ["Food"] * 8 +
    ["Transport"] * 7 +
    ["Shopping"] * 6 +
    ["Bills"] * 5 +
    ["Entertainment"] * 5 +
    ["Health"] * 4 +
    ["Education"] * 6
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_texts)

model = LogisticRegression(max_iter=1000)
model.fit(X, training_categories)


def predict_category(description):
    text = vectorizer.transform([description.lower()])
    return model.predict(text)[0]


# ============================================================
# SPENDING PREDICTION
# ============================================================

def predict_future_spending(expenses):

    if len(expenses) < 3:
        return None

    X = [[i + 1] for i in range(len(expenses))]
    y = expenses

    forecast_model = LinearRegression()
    forecast_model.fit(X, y)

    prediction = forecast_model.predict(
        [[len(expenses) + 1]]
    )[0]

    return max(0, round(float(prediction), 2))


# ============================================================
# DATABASE
# ============================================================

def get_db():

    conn = sqlite3.connect("finance.db")
    conn.row_factory = sqlite3.Row

    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            date TEXT NOT NULL,

            description TEXT NOT NULL,

            amount REAL NOT NULL,

            transaction_type TEXT NOT NULL,

            category TEXT NOT NULL

        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS budgets (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            category TEXT NOT NULL,

            amount REAL NOT NULL

        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# MAIN HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

    <title>AI Finance Assistant</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


    <style>

        * {
            box-sizing: border-box;
        }


        body {

            margin: 0;

            font-family: Arial, sans-serif;

            background: #f4f6f8;

            color: #222;

        }


        header {

            background: #17202a;

            color: white;

            padding: 25px;

            text-align: center;

        }


        header h1 {

            margin: 0;

            font-size: 30px;

        }


        header p {

            margin-bottom: 0;

            color: #d5d8dc;

        }


        .container {

            width: 92%;

            max-width: 1200px;

            margin: 25px auto;

        }


        .cards {

            display: grid;

            grid-template-columns:
                repeat(3, 1fr);

            gap: 18px;

        }


        .card {

            background: white;

            padding: 22px;

            border-radius: 12px;

            box-shadow:
                0 2px 10px rgba(0,0,0,0.08);

        }


        .card h3 {

            margin-top: 0;

            color: #566573;

        }


        .card h2 {

            margin-bottom: 0;

        }


        .income {

            color: #188038;

        }


        .expense {

            color: #d93025;

        }


        .section {

            background: white;

            margin-top: 22px;

            padding: 25px;

            border-radius: 12px;

            box-shadow:
                0 2px 10px rgba(0,0,0,0.08);

        }


        .section h2 {

            margin-top: 0;

        }


        input,
        select {

            padding: 11px;

            margin: 5px;

            border: 1px solid #ccd1d1;

            border-radius: 6px;

        }


        button {

            padding: 11px 18px;

            border: none;

            border-radius: 6px;

            background: #17202a;

            color: white;

            cursor: pointer;

        }


        button:hover {

            background: #34495e;

        }


        .delete {

            background: #c0392b;

        }


        .edit {

            background: #2874a6;

        }


        table {

            width: 100%;

            border-collapse: collapse;

        }


        th,
        td {

            padding: 12px;

            border-bottom:
                1px solid #e5e7e9;

            text-align: left;

        }


        th {

            background: #f2f3f4;

        }


        .ai-box {

            background: #ebf5fb;

            padding: 16px;

            border-left:
                5px solid #3498db;

            margin-top: 15px;

            border-radius: 5px;

        }


        .prediction {

            background: #f5eef8;

            border-left:
                5px solid #8e44ad;

        }


        .insight {

            background: #eafaf1;

            border-left:
                5px solid #27ae60;

        }


        .budget {

            padding: 12px;

            margin: 8px 0;

            background: #f8f9f9;

            border-radius: 6px;

        }


        .small {

            color: #777;

            font-size: 13px;

        }


        @media(max-width: 700px) {

            .cards {

                grid-template-columns: 1fr;

            }

            table {

                font-size: 13px;

            }

            input,
            select {

                width: 100%;

                margin: 5px 0;

            }

            button {

                margin-top: 5px;

            }

        }

    </style>

</head>


<body>


<header>

    <h1>AI Finance Assistant</h1>

    <p>
        Personal Finance Management
        using Machine Learning
    </p>

</header>


<div class="container">


    <!-- ================================================= -->
    <!-- DASHBOARD -->
    <!-- ================================================= -->

    <div class="cards">


        <div class="card">

            <h3>Total Income</h3>

            <h2 class="income">
                ₹{{ "%.2f"|format(income) }}
            </h2>

        </div>


        <div class="card">

            <h3>Total Expenses</h3>

            <h2 class="expense">
                ₹{{ "%.2f"|format(expense) }}
            </h2>

        </div>


        <div class="card">

            <h3>Current Balance</h3>

            <h2>
                ₹{{ "%.2f"|format(balance) }}
            </h2>

        </div>


    </div>


    <!-- ================================================= -->
    <!-- ADD TRANSACTION -->
    <!-- ================================================= -->

    <div class="section">

        <h2>Add Transaction</h2>


        <form method="POST"
              action="/add">


            <input

                type="date"

                name="date"

                value="{{ today }}"

                required>


            <input

                type="text"

                name="description"

                placeholder="Example: Swiggy dinner"

                required>


            <input

                type="number"

                name="amount"

                placeholder="Amount"

                step="0.01"

                min="0"

                required>


            <select

                name="transaction_type"

                required>

                <option value="Expense">
                    Expense
                </option>

                <option value="Income">
                    Income
                </option>

            </select>


            <select name="category">

                <option value="">
                    AI Predict Category
                </option>

                <option>Food</option>

                <option>Transport</option>

                <option>Shopping</option>

                <option>Bills</option>

                <option>Entertainment</option>

                <option>Health</option>

                <option>Education</option>

                <option>Other</option>

            </select>


            <button type="submit">

                Add Transaction

            </button>


        </form>


        <div class="ai-box">

            <b>AI Expense Categorization</b>

            <p>

                The ML model analyzes the
                transaction description and
                predicts its category.

            </p>

            <p>

                Example:

                <b>Swiggy dinner → Food</b>

            </p>

        </div>

    </div>


    <!-- ================================================= -->
    <!-- BUDGET -->
    <!-- ================================================= -->

    <div class="section">

        <h2>Monthly Budget</h2>


        <form method="POST"
              action="/budget">


            <select
                name="category"
                required>

                <option value="">
                    Select Category
                </option>

                <option>Food</option>

                <option>Transport</option>

                <option>Shopping</option>

                <option>Bills</option>

                <option>Entertainment</option>

                <option>Health</option>

                <option>Education</option>

            </select>


            <input

                type="number"

                name="amount"

                placeholder="Budget Amount"

                step="0.01"

                min="0"

                required>


            <button type="submit">

                Set Budget

            </button>


        </form>


        {% for budget in budgets %}

            <div class="budget">

                <b>
                    {{ budget["category"] }}
                </b>

                :

                ₹{{ "%.2f"|format(budget["amount"]) }}

            </div>

        {% endfor %}

    </div>


    <!-- ================================================= -->
    <!-- SPENDING CHART -->
    <!-- ================================================= -->

    <div class="section">

        <h2>Spending Analysis</h2>

        <canvas id="expenseChart"></canvas>

    </div>


    <!-- ================================================= -->
    <!-- SPENDING PREDICTION -->
    <!-- ================================================= -->

    <div class="section prediction">

        <h2>
            🔮 Spending Prediction
        </h2>


        {% if predicted_spending is not none %}

            <p>

                Based on your previous
                expense pattern, the ML model
                estimates the next spending
                amount as:

            </p>


            <h2>

                ₹{{ "%.2f"|format(predicted_spending) }}

            </h2>


        {% else %}

            <p>

                Add at least 3 expense
                transactions to generate
                a spending prediction.

            </p>

        {% endif %}


        <p class="small">

            This is a basic ML-based
            educational estimate.

        </p>

    </div>


    <!-- ================================================= -->
    <!-- AI INSIGHTS -->
    <!-- ================================================= -->

    <div class="section insight">

        <h2>
            🤖 AI Financial Insights
        </h2>


        {% if expense > 0 %}


            <p>

                Your total spending is

                <b>
                    ₹{{ "%.2f"|format(expense) }}
                </b>.

            </p>


            {% if highest_category %}

                <p>

                    Your highest spending
                    category is

                    <b>
                        {{ highest_category }}
                    </b>

                    with

                    <b>
                        ₹{{ "%.2f"|format(highest_amount) }}
                    </b>.

                </p>

            {% endif %}


            {% if income > 0 and expense > income %}

                <p class="expense">

                    ⚠ Your expenses are
                    currently higher than
                    your income.

                </p>


                <p>

                    Suggestion: Review your
                    high-spending categories
                    and consider reducing
                    unnecessary expenses.

                </p>


            {% elif income > 0 %}

                <p class="income">

                    ✓ Your current expenses
                    are within your income.

                </p>


                <p>

                    Suggestion: Continue
                    monitoring your spending
                    and stay within your
                    monthly budget.

                </p>

            {% endif %}


        {% else %}

            <p>

                Add expense transactions
                to generate AI insights.

            </p>

        {% endif %}


        <p class="small">

            Educational information only.
            Not professional financial,
            tax, or investment advice.

        </p>

    </div>


    <!-- ================================================= -->
    <!-- TRANSACTION HISTORY -->
    <!-- ================================================= -->

    <div class="section">

        <h2>
            Transaction History
        </h2>


        {% if transactions %}


        <table>

            <tr>

                <th>Date</th>

                <th>Description</th>

                <th>Amount</th>

                <th>Type</th>

                <th>Category</th>

                <th>Action</th>

            </tr>


            {% for t in transactions %}


            <tr>

                <td>
                    {{ t["date"] }}
                </td>


                <td>
                    {{ t["description"] }}
                </td>


                <td>
                    ₹{{ "%.2f"|format(t["amount"]) }}
                </td>


                <td>

                    {% if t["transaction_type"] == "Income" %}

                        <span class="income">
                            Income
                        </span>

                    {% else %}

                        <span class="expense">
                            Expense
                        </span>

                    {% endif %}

                </td>


                <td>
                    {{ t["category"] }}
                </td>


                <td>

                    <a href="/delete/{{ t['id'] }}">

                        <button
                            class="delete"
                            type="button">

                            Delete

                        </button>

                    </a>

                </td>

            </tr>


            {% endfor %}


        </table>


        {% else %}

            <p>
                No transactions yet.
            </p>

        {% endif %}

    </div>


    <!-- ================================================= -->
    <!-- PROJECT INFORMATION -->
    <!-- ================================================= -->

    <div class="section">

        <h2>
            About the Project
        </h2>


        <p>

            <b>AI Finance Assistant</b>
            is a beginner-friendly
            personal finance management
            application.

        </p>


        <p>

            It uses Machine Learning
            for expense categorization
            and basic spending prediction.

        </p>


        <p>

            Technologies:

            Python,
            Flask,
            SQLite,
            Scikit-learn,
            HTML,
            CSS,
            JavaScript,
            Chart.js.

        </p>

    </div>


</div>


<!-- ===================================================== -->
<!-- CHART SCRIPT -->
<!-- ===================================================== -->

<script>

const categories =
    {{ chart_categories | safe }};

const amounts =
    {{ chart_amounts | safe }};


new Chart(

    document.getElementById(
        "expenseChart"
    ),

    {

        type: "bar",

        data: {

            labels: categories,

            datasets: [

                {

                    label:
                        "Spending by Category",

                    data: amounts

                }

            ]

        },

        options: {

            responsive: true,

            plugins: {

                legend: {

                    display: true

                }

            }

        }

    }

);

</script>


</body>

</html>

"""


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def index():

    conn = get_db()


    transactions = conn.execute(

        """
        SELECT *
        FROM transactions
        ORDER BY id DESC
        """

    ).fetchall()


    budgets = conn.execute(

        """
        SELECT *
        FROM budgets
        """

    ).fetchall()


    income_row = conn.execute(

        """
        SELECT SUM(amount)
        FROM transactions
        WHERE transaction_type = 'Income'
        """

    ).fetchone()


    expense_row = conn.execute(

        """
        SELECT SUM(amount)
        FROM transactions
        WHERE transaction_type = 'Expense'
        """

    ).fetchone()


    income = income_row[0] or 0

    expense = expense_row[0] or 0

    balance = income - expense


    category_rows = conn.execute(

        """
        SELECT category,
               SUM(amount) AS total

        FROM transactions

        WHERE transaction_type = 'Expense'

        GROUP BY category

        ORDER BY total DESC
        """

    ).fetchall()


    expense_rows = conn.execute(

        """
        SELECT amount

        FROM transactions

        WHERE transaction_type = 'Expense'

        ORDER BY id
        """

    ).fetchall()


    expense_values = [

        row["amount"]

        for row in expense_rows

    ]


    predicted_spending = \
        predict_future_spending(
            expense_values
        )


    highest_category = None

    highest_amount = 0


    if category_rows:

        highest = max(

            category_rows,

            key=lambda row:
                row["total"]

        )

        highest_category = \
            highest["category"]

        highest_amount = \
            highest["total"]


    conn.close()


    chart_categories = [

        row["category"]

        for row in category_rows

    ]


    chart_amounts = [

        row["total"]

        for row in category_rows

    ]


    return render_template_string(

        HTML,

        transactions=transactions,

        budgets=budgets,

        income=income,

        expense=expense,

        balance=balance,

        today=date.today(),

        chart_categories=
            chart_categories,

        chart_amounts=
            chart_amounts,

        predicted_spending=
            predicted_spending,

        highest_category=
            highest_category,

        highest_amount=
            highest_amount

    )


# ============================================================
# ADD TRANSACTION
# ============================================================

@app.route(
    "/add",
    methods=["POST"]
)
def add_transaction():

    transaction_date = \
        request.form["date"]

    description = \
        request.form["description"]

    amount = \
        float(request.form["amount"])

    transaction_type = \
        request.form["transaction_type"]

    category = \
        request.form.get("category")


    if transaction_type == "Expense":

        if not category:

            category = \
                predict_category(
                    description
                )

    else:

        category = "Income"


    conn = get_db()


    conn.execute(

        """
        INSERT INTO transactions
        (
            date,
            description,
            amount,
            transaction_type,
            category
        )

        VALUES (?, ?, ?, ?, ?)
        """,

        (
            transaction_date,
            description,
            amount,
            transaction_type,
            category
        )

    )


    conn.commit()

    conn.close()


    return redirect("/")


# ============================================================
# DELETE TRANSACTION
# ============================================================

@app.route(
    "/delete/<int:id>"
)
def delete_transaction(id):

    conn = get_db()


    conn.execute(

        """
        DELETE FROM transactions
        WHERE id = ?
        """,

        (id,)

    )


    conn.commit()

    conn.close()


    return redirect("/")


# ============================================================
# SET BUDGET
# ============================================================

@app.route(
    "/budget",
    methods=["POST"]
)
def set_budget():

    category = \
        request.form["category"]

    amount = \
        float(request.form["amount"])


    conn = get_db()


    conn.execute(

        """
        INSERT INTO budgets
        (category, amount)

        VALUES (?, ?)
        """,

        (
            category,
            amount
        )

    )


    conn.commit()

    conn.close()


    return redirect("/")


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )