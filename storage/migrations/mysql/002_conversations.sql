CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    source VARCHAR(50),
    turn_count INT NOT NULL DEFAULT 1,
    total_cost DECIMAL(10, 6) NOT NULL DEFAULT 0,
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    organization_id VARCHAR(255),
    user_id VARCHAR(255),
    application_id VARCHAR(255),
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_conversations_session (session_id),
    INDEX idx_conversations_org (organization_id),
    INDEX idx_conversations_user (user_id),
    INDEX idx_conversations_created (created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(36) PRIMARY KEY,
    conversation_id VARCHAR(36) NOT NULL,
    layer ENUM('thread', 'llm_context', 'trace') NOT NULL,
    turn_number INT NOT NULL,
    sequence_order INT NOT NULL DEFAULT 0,
    role ENUM('user', 'assistant', 'system', 'tool') NOT NULL,
    content TEXT NOT NULL,
    model VARCHAR(255),
    input_tokens INT,
    output_tokens INT,
    cost DECIMAL(10, 6),
    node_name VARCHAR(255),
    capabilities JSON,
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_messages_conversation (conversation_id),
    INDEX idx_messages_layer (conversation_id, layer),
    INDEX idx_messages_turn (conversation_id, turn_number),
    CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS message_feedback (
    id VARCHAR(36) PRIMARY KEY,
    message_id VARCHAR(36) NOT NULL,
    feedback_type ENUM('thumbs_up', 'thumbs_down', 'remark', 'case'),
    status ENUM('open', 'reviewed', 'resolved', 'closed') NOT NULL DEFAULT 'open',
    content TEXT,
    score INT,
    created_by VARCHAR(255),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_message_feedback_message (message_id),
    INDEX idx_message_feedback_type (feedback_type),
    CONSTRAINT fk_feedback_message FOREIGN KEY (message_id) REFERENCES messages(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS llm_context_refs (
    conversation_id VARCHAR(36) NOT NULL,
    turn_number INT NOT NULL,
    message_id VARCHAR(36) NOT NULL,
    sequence_order INT NOT NULL DEFAULT 0,
    PRIMARY KEY (conversation_id, turn_number, message_id),
    INDEX idx_llm_refs_conversation (conversation_id),
    INDEX idx_llm_refs_turn (conversation_id, turn_number),
    CONSTRAINT fk_llm_refs_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(id),
    CONSTRAINT fk_llm_refs_message FOREIGN KEY (message_id) REFERENCES messages(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
