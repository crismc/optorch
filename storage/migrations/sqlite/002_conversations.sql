CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source TEXT,
    turn_count INTEGER NOT NULL DEFAULT 1,
    total_cost REAL NOT NULL DEFAULT 0,
    currency TEXT NOT NULL DEFAULT 'USD',
    organization_id TEXT,
    user_id TEXT,
    application_id TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE (session_id)
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_org ON conversations(organization_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);

CREATE TRIGGER IF NOT EXISTS conversations_updated_at
    AFTER UPDATE ON conversations
    FOR EACH ROW
BEGIN
    UPDATE conversations SET updated_at = datetime('now') WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    layer TEXT NOT NULL CHECK (layer IN ('thread', 'llm_context', 'trace')),
    turn_number INTEGER NOT NULL,
    sequence_order INTEGER NOT NULL DEFAULT 0,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost REAL,
    node_name TEXT,
    capabilities TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_layer ON messages(conversation_id, layer);
CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(conversation_id, turn_number);

CREATE TABLE IF NOT EXISTS message_feedback (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id),
    feedback_type TEXT CHECK (feedback_type IN ('thumbs_up', 'thumbs_down', 'remark', 'case')),
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'reviewed', 'resolved', 'closed')),
    content TEXT,
    score INTEGER,
    created_by TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_message_feedback_message ON message_feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_message_feedback_type ON message_feedback(feedback_type);

CREATE TABLE IF NOT EXISTS llm_context_refs (
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    turn_number INTEGER NOT NULL,
    message_id TEXT NOT NULL REFERENCES messages(id),
    sequence_order INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (conversation_id, turn_number, message_id)
);

CREATE INDEX IF NOT EXISTS idx_llm_refs_conversation ON llm_context_refs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_llm_refs_turn ON llm_context_refs(conversation_id, turn_number);
