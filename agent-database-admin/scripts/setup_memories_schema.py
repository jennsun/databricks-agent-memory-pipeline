"""Create the "memories" schema and grant the app's SP full access."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from databricks_ai_bridge.lakebase import (
    LakebaseClient,
    SchemaPrivilege,
    SequencePrivilege,
    TablePrivilege,
)


SP_CLIENT_ID = "<your-app-service-principal-client-id>"
MEMORY_SCHEMA = os.getenv("LAKEBASE_AGENT_MEMORY_SCHEMA", "memories")
PROJECT = "<your-agent-memory-project>"
BRANCH = "<your-agent-memory-branch>"


def main():
    print(f"Connecting to Lakebase project={PROJECT} branch={BRANCH}")
    with LakebaseClient(project=PROJECT, branch=BRANCH) as client:
        print(f"Creating role for SP {SP_CLIENT_ID}...")
        try:
            client.create_role(SP_CLIENT_ID, "SERVICE_PRINCIPAL")
            print("  Role created.")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("  Role already exists.")
            else:
                raise

        for schema_to_create in [MEMORY_SCHEMA, "agent_server"]:
            print(f"Creating schema '{schema_to_create}'...")
            client.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema_to_create}"')
            print(f"  Schema '{schema_to_create}' ready.")

        schemas = [MEMORY_SCHEMA, "agent_server", "ai_chatbot", "drizzle"]
        for schema in schemas:
            print(f"Granting USAGE + CREATE on schema '{schema}'...")
            try:
                client.grant_schema(
                    grantee=SP_CLIENT_ID,
                    schemas=[schema],
                    privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
                )
            except Exception as e:
                print(f"  Warning: {e}")

            print(f"Granting all-tables privileges in '{schema}'...")
            try:
                client.grant_all_tables_in_schema(
                    grantee=SP_CLIENT_ID,
                    schemas=[schema],
                    privileges=[
                        TablePrivilege.SELECT,
                        TablePrivilege.INSERT,
                        TablePrivilege.UPDATE,
                        TablePrivilege.DELETE,
                    ],
                )
            except Exception as e:
                print(f"  Warning: {e}")

            print(f"Granting all-sequences privileges in '{schema}'...")
            try:
                client.grant_all_sequences_in_schema(
                    grantee=SP_CLIENT_ID,
                    schemas=[schema],
                    privileges=[
                        SequencePrivilege.USAGE,
                        SequencePrivilege.SELECT,
                        SequencePrivilege.UPDATE,
                    ],
                )
            except Exception as e:
                print(f"  Warning: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
