from crewai import Agent, Crew, Task
import json

# Load all matching product entries from database
def load_product_details(product_id: str) -> list:
    with open("product_database.json", "r") as f:
        products = json.load(f)
    return [
        p for p in products
        if p.get("product_id", "").strip().lower() == product_id.strip().lower()
    ]

class MarketplaceCrew:
    def __init__(self, input_data, verbose=True):
        self.verbose = verbose
        self.input_data = input_data
        self.crew = None
        self.result = None
        self.setup()

    def setup(self):
        # Step 1: Handle unstructured input by parsing
        if isinstance(self.input_data, str):
            parser_agent = Agent(
                role="Buyer Intent Parser",
                goal="Understand buyer's request and convert it into structured product request",
                backstory="You are an expert assistant that turns user sentences into backend-ready JSON.",
                allow_delegation=False,
                verbose=self.verbose
            )

            parsing_task = Task(
                description=(
                    "Parse the following user request into a valid JSON object with keys: "
                    "ProductID, Quantity, Budget, Instruction.\n\n"
                    f"Input:\n{self.input_data}\n\n"
                    "Requirements:\n"
                    "- ProductID must be a string like 'p5'.\n"
                    "- Quantity must be an integer.\n"
                    "- Budget must be a float.\n"
                    "- Instruction is a sentence or phrase expressing preference.\n\n"
                    "Return ONLY the JSON object with these 4 keys."
                ),
                expected_output="A valid JSON object with keys: ProductID, Quantity, Budget, Instruction.",
                agent=parser_agent
            )

            parsed = Crew(agents=[parser_agent], tasks=[parsing_task]).kickoff()

            try:
                self.input_data = json.loads(str(parsed).strip())
            except Exception as e:
                self.result = f" Failed to parse input: {e}\nRaw output:\n{parsed}"
                return
        
        # Step 2: Validate and cast values
        try:
            self.input_data["Budget"] = float(self.input_data["Budget"])
            
            self.input_data["Quantity"] = int(self.input_data["Quantity"])
        except (ValueError, TypeError, KeyError) as e:
            self.result = (
                f" Invalid input format: {e} "
                f"(Budget: {self.input_data.get('Budget')}, Quantity: {self.input_data.get('Quantity')})"
            )


        try:
            product_id = self.input_data["ProductID"]  # throws KeyError if missing
            product_details = load_product_details(product_id)

            if not product_details:
                raise ValueError(f"No product details found for ProductID: {product_id}")

            self.input_data["ProductDetails"] = product_details

        except KeyError:
            self.result = " 'ProductID' key missing in input data."
            return

        except ValueError as e:
            self.result = f" {e}"
            return

        except Exception as e:
            self.result = f" Unexpected error while loading product details: {e}"
            return

        if not product_details:
            self.result = (
                f" Agent: Sorry, no matches found for ProductID '{self.input_data['ProductID']}'."
            )
            return

        self.input_data["ProductDetails"] = product_details
        self.crew = self.create_crew()

    def create_crew(self):
        # Junior Seller Agent
        junior_seller = Agent(
            role="Junior Seller Agent",
            goal="Shortlist up to 3 matching sellers",
            backstory="Expert in filtering offers based on constraints like budget and instruction.",
            allow_delegation=False,
            verbose=self.verbose
        )

        junior_seller_task = Task(
            description=(
                f"You are given a product request and a list of sellers:\n\n"
                f"• Product Details: {self.input_data['ProductDetails']}\n"
                f"• Quantity: {self.input_data['Quantity']}\n"
                f"• Unit Budget: {self.input_data['Budget']:.2f}\n"
                f"• Buyer Instruction: {self.input_data['Instruction']}\n\n"
                "Select up to 3 sellers where:\n"
                "- product_id matches\n"
                "- unit price ≤ unit budget\n"
                "- description meets instruction\n"
                "- quantity is available\n\n"
                "Return JSON list: SellerID, ProductID, Price, Description, Reason.\n"
                "If no match, say: 'No suitable sellers available.'"
            ),
            expected_output="List of up to 3 sellers or a message if none match.",
            agent=junior_seller
        )

        # Senior Seller Agent
        senior_seller = Agent(
            role="Senior Seller Review Agent",
            goal="Validate or correct the junior seller shortlist",
            backstory="You're the QA lead ensuring seller options meet buyer constraints.",
            allow_delegation=False,
            verbose=self.verbose
        )

        senior_seller_task = Task(
            description=(
                f"Review the seller shortlist for ProductID={self.input_data['ProductID']}:\n"
                f"- Confirm unit price ≤ {self.input_data['Budget']:.2f}\n"
                "- Ensure instruction is met\n"
                "- Confirm quantity is sufficient\n"
                "If invalid, revise and give reason.\n"
                "If none qualify, say: 'No valid sellers.'"
            ),
            expected_output="Final shortlist or rejection with feedback.",
            agent=senior_seller
        )

        # Junior Buyer Agent
        junior_buyer = Agent(
            role="Junior Buyer Agent",
            goal="Select the best seller option",
            backstory="You evaluate cost and instruction fit to choose the most suitable seller.",
            allow_delegation=False,
            verbose=self.verbose
        )

        junior_buyer_task = Task(
            description=(
                "Pick the best seller from the senior-approved list:\n"
                "- Lowest acceptable price\n"
                "- Matches instruction well\n"
                "- Quantity available\n\n"
                "Return JSON: SellerID, Price, Description, Reason."
            ),
            expected_output="Best seller selected with reason.",
            agent=junior_buyer
        )

        # Senior Buyer Agent
        senior_buyer = Agent(
            role="Senior Buyer Review Agent",
            goal="Approve or revise the final choice",
            backstory="You verify that the chosen seller fits all constraints.",
            allow_delegation=False,
            verbose=self.verbose
        )

        senior_buyer_task = Task(
            description=(
                f"Review the chosen seller:\n"
                f"- Ensure price ≤ {self.input_data['Budget']:.2f}\n"
                "- Confirm description matches instruction\n"
                "- Confirm quantity meets request\n"
                "Approve if valid, else revise and explain."
            ),
            expected_output="Final approval and reasoning in JSON.",
            agent=senior_buyer
        )

        crew = Crew(
            agents=[junior_seller, senior_seller, junior_buyer, senior_buyer],
            tasks=[junior_seller_task, senior_seller_task, junior_buyer_task, senior_buyer_task]
        )
        return crew

    def run(self):
        if self.result:
            return self.result
        if self.crew:
            return self.crew.kickoff()
        return " Unexpected error: crew could not be initialized."
